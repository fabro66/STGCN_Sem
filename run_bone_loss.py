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
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from tool.utils import deterministic_random

args = parse_args()
print(args)

try:
    # Create checkpoint direction if it does not exit
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError("Unable to create checkpoint direction:", args.checkpoint)

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
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                               filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)

model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
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

# geometric constrain
edge_children = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])
edge_parent = np.array([0, 1, 2, 0, 4, 5, 8, 11, 12, 8, 14, 15])
edge_matrix = np.zeros((len(edge_children), 17), dtype=np.float32)
for i in range(len(edge_children)):
    edge_matrix[i][edge_children[i]] = 1.0
    edge_matrix[i][edge_parent[i]] = -1.0
edge_matrix = torch.from_numpy(edge_matrix)

bone_children = np.array([3, 4, 5, 9, 10, 11])
bone_parent = np.array([0, 1, 2, 6, 7, 8])
bone_matrix = np.zeros((len(bone_children), 12), dtype=np.float32)
for i in range(len(bone_children)):
    bone_matrix[i][bone_children[i]] = 1.0
    bone_matrix[i][bone_parent[i]] = -1.0
bone_matrix = torch.from_numpy(bone_matrix)

weight_bone = torch.Tensor([0.2])

if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_bone_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []
    losses_bone_valid = []

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

    max_norm = True
    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_bone_loss = 0
        N = 0
        model_pos_train.train()

        # Regular supervised scenario
        for _, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

                edge_matrix = edge_matrix.cuda()
                bone_matrix = bone_matrix.cuda()
                weight_bone = weight_bone.cuda()

            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_pos_train(inputs_2d)

            # geometric constrain
            bone_length = torch.matmul(edge_matrix, predicted_3d_pos)
            bone_length = bone_length.pow(2)
            bone_length = bone_length.sum(-1)
            bone_length = torch.sqrt(bone_length)
            bone_length = bone_length.unsqueeze(dim=-1)

            loss_bone = torch.matmul(bone_matrix, bone_length)
            loss_bone = torch.abs(loss_bone)
            loss_bone_sum = torch.sum(loss_bone, dim=-2)
            loss_bone_mean = torch.mean(loss_bone_sum)

            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)

            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            epoch_bone_loss += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_bone_mean

            loss_total = loss_3d_pos + weight_bone*loss_bone_mean
            loss_total.backward()

            # Control the grad of parameters
            # if max_norm:
            #     nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)
        losses_bone_train.append(epoch_bone_loss / N)

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
                    predicted_3d_pos = model_pos(inputs_2d)
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
                    inputs_traj = inputs_3d[:, :, :1].clone()
                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f bone_loss %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_bone_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f bone_loss %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses_3d_train[-1] * 1000,
                    losses_3d_train_eval[-1] * 1000,
                    losses_3d_valid[-1] * 1000,
                    losses_bone_train[-1] * 1000))

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
    epoch_losses_bone = 0

    # geometric constrain
    edge_children = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])
    edge_parent = np.array([0, 1, 2, 0, 4, 5, 8, 11, 12, 8, 14, 15])
    edge_matrix = np.zeros((len(edge_children), 17), dtype=np.float32)
    for i in range(len(edge_children)):
        edge_matrix[i][edge_children[i]] = 1.0
        edge_matrix[i][edge_parent[i]] = -1.0
    edge_matrix = torch.from_numpy(edge_matrix)

    bone_children = np.array([3, 4, 5, 9, 10, 11])
    bone_parent = np.array([0, 1, 2, 6, 7, 8])
    bone_matrix = np.zeros((len(bone_children), 12), dtype=np.float32)
    for i in range(len(bone_children)):
        bone_matrix[i][bone_children[i]] = 1.0
        bone_matrix[i][bone_parent[i]] = -1.0
    bone_matrix = torch.from_numpy(bone_matrix)

    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

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
                edge_matrix = edge_matrix.cuda()
                bone_matrix = bone_matrix.cuda()

            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            # geometric constrain
            bone_length = torch.matmul(edge_matrix, predicted_3d_pos)
            bone_length = bone_length.pow(2)
            bone_length = bone_length.sum(-1)
            bone_length = torch.sqrt(bone_length)
            bone_length = bone_length.unsqueeze(dim=-1)

            loss_bone = torch.matmul(bone_matrix, bone_length)
            loss_bone = torch.abs(loss_bone)
            loss_bone_sum = torch.sum(loss_bone, dim=-2)
            loss_bone_mean = torch.mean(loss_bone_sum)

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
                                                                                         inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            epoch_losses_bone += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_bone_mean.item()

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
    bone_loss = (epoch_losses_bone / N) * 1000

    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('Bone error: ', bone_loss, 'mm')
    print('----------')

    return e1, e2, e3, ev, bone_loss


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

        from common.visualization import render_animation

        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)

else:
    print('Evaluating...')
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
        errors_bone = []

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
            e1, e2, e3, ev, bone_loss = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)
            errors_bone.append(bone_loss)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')
        print('Bone Error            action-wise average:', round(np.mean(bone_loss), 2), 'mm')


    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')
