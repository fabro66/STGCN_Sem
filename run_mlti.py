import numpy as np
from collections import OrderedDict

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.graph_utils import adj_mx_from_skeleton
from common.camera import *
from model.stgcn import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from tool.utils import deterministic_random

args = parse_args()
print(args)

try:
    # Create checkpoint direction if it does not exict
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError("Unable to create checkpoint direction:", args.checkpoint)

file_source = sys.argv[0]
dir_file = os.path.dirname(file_source)
log_file = os.path.join(dir_file, args.checkpoint, "log.txt")
with open(log_file, "w") as fw:
    fw.write(" {}".format(args))

print("Loading dataset...")
dataset_path = "data/data_3d_" + args.dataset + ".npz"
if args.dataset == "h36m":
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
else:
    raise KeyError("Invalid dataset")

print("Preparing data...")
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if "positions" in anim:
            positions_3d = []
            for cam in anim["cameras"]:
                pos_3d = world_to_camera(anim["positions"], R=cam["orientation"], t=cam["translation"])
                pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim["positions_3d"] = positions_3d

print("Loading 2D detections...")
keypoints = np.load("data/data_2d_" + args.dataset + "_" + args.keypoints + ".npz", allow_pickle=True)
keypoints_metadata = keypoints["metadata"].item()
keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints["positions_2d"].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
        if "positions_3d" not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]["positions_3d"])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam["res_w"], h=cam["res_h"])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args.subjects_train.split(',')
if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    subjects_test = [args.viz_subject]


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

filter_widths = [int(x) for x in args.architecture.split(",")]
adj = adj_mx_from_skeleton(dataset.skeleton())
if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pos_train = SpatialTemporalModel(adj, poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                                      dataset.skeleton().num_joints(), filter_widths=filter_widths,
                                                      causal=args.causal, dropout=args.dropout, channels=args.channels)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos_train = SpatialTemporalModel(adj, poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                           dataset.skeleton().num_joints(), filter_widths=filter_widths,
                                           causal=args.causal, dropout=args.dropout, channels=args.channels, dense=args.dense)

model_pos = SpatialTemporalModel(adj, poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                 filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)

receptive_field = model_pos.receptive_field()
print("INFO: Receptive field: {} frames".format(receptive_field))
pad = (receptive_field - 1) // 2  # padding on each side
if args.causal:
    print("INFO: Using causal convolutions")
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print("INFO: Trainable parameter count: ", model_params)

with open(log_file, "a") as fw:
    fw.write("\nINFO: Trainable parameter count: {}".format(model_params))

mutil_gpu = False
cpu = False
gpu = False
# Multi-gpu training
if torch.cuda.device_count() > 1:
    print("The number of GPU: {}".format(torch.cuda.device_count()))
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
    model_pos = nn.DataParallel(model_pos, device_ids=[0, 1])
    model_pos_train = nn.DataParallel(model_pos_train, device_ids=[0, 1])
    mutil_gpu = True
elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
    gpu = True
else:
    cpu = True

if args.resume or args.evaluate:
    if mutil_gpu or gpu:
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
        print("Loading checkpoint", chk_filename)
        checkpoint = torch.load(chk_filename)
        print("This model was trained for {} epochs".format(checkpoint["epoch"]))
        model_pos_train.load_state_dict(checkpoint["model_pos"])
        model_pos.load_state_dict(checkpoint["model_pos"])
    else:
        new_state_dict = OrderedDict()
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
        print("Loading checkpoint", chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print("This model was trained for {} epochs".format(checkpoint["epoch"]))

        for k, v in checkpoint["model_pos"].items():
            name = k[7:]  # remove "module"
            new_state_dict[name] = v
        model_pos_train.load_state_dict(new_state_dict)
        model_pos.load_state_dict(new_state_dict)

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print("INFO: Testing on {} frames".format(test_generator.num_frames()))

if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_v0 = []
    losses_3d_train_v1 = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_generator = ChunkedGenerator(args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d,
                                       args.stride,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                       joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')

    # max_norm = True
    loss_min = 49.5
    weight_loss = torch.Tensor([0.5])
    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_3d_train_v0 = 0
        epoch_loss_3d_train_v1 = 0
        N = 0
        model_pos_train.train()

        # Regular supervised scenario
        for _, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                weight_loss = weight_loss.cuda()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            pose_v0, pose_v1, predicted_3d_pos = model_pos_train(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            loss_3d_v0 = mpjpe(pose_v0, inputs_3d)
            loss_3d_v1 = mpjpe(pose_v1, inputs_3d)

            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            epoch_loss_3d_train_v0 += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_v0.item()
            epoch_loss_3d_train_v1 += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_v1.item()

            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_total = loss_3d_pos + weight_loss*loss_3d_v0 + weight_loss*loss_3d_v1
            loss_total.backward()

            # Control the grad of parameters
            # if max_norm:
            #     nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)
        losses_3d_train_v0.append(epoch_loss_3d_train_v0 / N)
        losses_3d_train_v1.append(epoch_loss_3d_train_v1 / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()

            epoch_loss_3d_valid = 0
            N = 0

            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()

                    inputs_3d[:, :, 0] = 0

                    # Predict 3D poses
                    _, _, predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    _, _, predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f 3d_train_v1 %f 3d_train_v0 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_v1[-1] * 1000,
                    losses_3d_train_v0[-1] * 1000))

            with open(log_file, "a") as fw:
                fw.write('\n[%d] time %.2f lr %f 3d_train %f 3d_train_v0 %f 3d_train_v1 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_v0[-1] * 1000,
                    losses_3d_train_v1[-1] * 1000))

        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_train_v1 %f 3d_train_v0 %f 3d_eval %f 3d_valid %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_v1[-1] * 1000,
                    losses_3d_train_v0[-1] * 1000,
                    losses_3d_train_eval[-1] * 1000,
                    losses_3d_valid[-1] * 1000))

            with open(log_file, "a") as fw:
                fw.write('\n[%d] time %.2f lr %f 3d_train 3d_train_v1 %f %f 3d_train_v0 %f 3d_eval %f 3d_valid %f' % (
                        epoch + 1,
                        elapsed,
                        lr,
                        losses_3d_train[-1] * 1000,
                        losses_3d_train_v1[-1] * 1000,
                        losses_3d_train_v0[-1] * 1000,
                        losses_3d_train_eval[-1] * 1000,
                        losses_3d_valid[-1] * 1000))

        # Saving the best result
        if losses_3d_valid[-1]*1000 < loss_min:
            chk_path = os.path.join(args.checkpoint, 'epoch_best.bin')
            print('Saving checkpoint to', chk_path)

            with open(log_file, "a") as fw:
                fw.write('\nSaving checkpoint to {}'.format(chk_path))

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict()
            }, chk_path)

            loss_min = losses_3d_valid[-1]*1000

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
        if torch.cuda.device_count() > 1:
            model_pos_train.module.set_bn_momentum(momentum)
        else:
            model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict()
            }, chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))
            plt.close('all')


# Evaluate
def evaluate(test_generator, action=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            _, _, predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
                                                                                         inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    with open(log_file, "a") as fw:
        fw.write('\n----' + action + '----')
        fw.write('\nTest time augmentation: {}'.format(test_generator.augment_enabled()))
        fw.write('\nProtocol #1 Error (MPJPE): {} mm'.format(e1))
        fw.write('\nProtocol #2 Error (P-MPJPE): {} mm'.format(e2))
        fw.write('\nProtocol #3 Error (N-MPJPE): {} mm'.format(e3))
        fw.write('\nVelocity Error (MPJVE): {} mm'.format(ev))
        fw.write('\n----------')

    return e1, e2, e3, ev


if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    prediction = evaluate(gen, return_predictions=True)

    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)
        # Predictions are in camera space
        np.save(args.viz_export, prediction)

    if args.viz_output is not None:
        if ground_truth is not None:
            # Reapply trajectory
            trajectory = ground_truth[:, :1]
            ground_truth[:, 1:] += trajectory
            prediction += trajectory

        # Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth

        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

        from tool.visualization import render_animation

        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)

else:
    print('Evaluating...')

    with open(log_file, "a") as fw:
        fw.write('\nEvaluating...')

    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))


    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)):  # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_poses_3d, out_poses_2d


    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right)
            e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')

        with open(log_file, "a") as fw:
            fw.write('\nProtocol #1   (MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p1), 1)))
            fw.write('\nProtocol #2 (P-MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p2), 1)))
            fw.write('\nProtocol #3 (N-MPJPE) action-wise average: {} mm'.format(round(np.mean(errors_p3), 1)))
            fw.write('\nVelocity      (MPJVE) action-wise average: {} mm'.format(round(np.mean(errors_vel), 2)))


    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)

            with open(log_file, "a") as fw:
                fw.write('\nEvaluating on subject {}'.format(subject))

            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')
