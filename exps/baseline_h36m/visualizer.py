import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.misc import expmap2rotmat_torch, find_indices_256, find_indices_srnn, rotmat2xyz_torch

import torch
import torch.utils.data as data
import pickle

class H36MEval(data.Dataset):
    def __init__(self, config, split_name, paired=True):
        super(H36MEval, self).__init__()
        self._split_name = split_name
        self._h36m_anno_dir = config.h36m_anno_dir
        self._actions = ["walking", "eating", "smoking", "discussion", "directions",
                        "greeting", "phoning", "posing", "purchases", "sitting",
                        "sittingdown", "takingphoto", "waiting", "walkingdog",
                        "walkingtogether"]

        self.h36m_motion_input_length =  config.motion.h36m_input_length
        self.h36m_motion_target_length =  config.motion.h36m_target_length
        self.used_joint_indexes = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
        
        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step
        self._h36m_files = self._get_h36m_files()
        self._file_length = len(self.data_idx)
        self.h36m_22seqs=[]
        for file in self.h36m_seqs:
            file = file[:, self.used_joint_indexes, :]
            self.h36m_22seqs.append(file)
        with open('h36m_eval_feature.pkl', 'rb') as f:
            self.h36m_files_feature = pickle.load(f)


    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._h36m_files)

    def _get_h36m_files(self):

        # create list
        seq_names = []

        seq_names += open(
            os.path.join(self._h36m_anno_dir.replace('h36m', ''), "h36m_test.txt"), 'r'
            ).readlines()

        self.h36m_seqs = []
        self.data_idx = []
        idx = 0
        for subject in seq_names:
            subject = subject.strip()
            for act in self._actions:
                filename0 = '{0}/{1}/{2}_{3}.txt'.format(self._h36m_anno_dir, subject, act, 1)
                filename1 = '{0}/{1}/{2}_{3}.txt'.format(self._h36m_anno_dir, subject, act, 2)
                poses0 = self._preprocess(filename0)
                poses1 = self._preprocess(filename1)

                self.h36m_seqs.append(poses0)
                self.h36m_seqs.append(poses1)

                num_frames0 = poses0.shape[0]
                num_frames1 = poses1.shape[0]

                fs_sel1, fs_sel2 = find_indices_256(num_frames0, num_frames1,
                                   self.h36m_motion_input_length + self.h36m_motion_target_length,
                                   input_n=self.h36m_motion_input_length)
                #fs_sel1, fs_sel2 = find_indices_srnn(num_frames0, num_frames1,
                #                   self.h36m_motion_input_length + self.h36m_motion_target_length,
                #                   input_n=self.h36m_motion_input_length)
                valid_frames0 = fs_sel1[:, 0]
                tmp_data_idx_1 = [idx] * len(valid_frames0)
                tmp_data_idx_2 = list(valid_frames0)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                valid_frames1 = fs_sel2[:, 0]
                tmp_data_idx_1 = [idx + 1] * len(valid_frames1)
                tmp_data_idx_2 = list(valid_frames1)
                self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                idx += 2

    def _preprocess(self, filename):
        info = open(filename, 'r').readlines()
        pose_info = []
        for line in info:
            line = line.strip().split(',')
            if len(line) > 0:
                pose_info.append(np.array([float(x) for x in line]))
        pose_info = np.array(pose_info)
        pose_info = pose_info.reshape(-1, 33, 3)
        pose_info[:, :2] = 0
        N = pose_info.shape[0]
        pose_info = pose_info.reshape(-1, 3)
        pose_info = expmap2rotmat_torch(torch.tensor(pose_info).float()).reshape(N, 33, 3, 3)[:, 1:]
        pose_info = rotmat2xyz_torch(pose_info)

        sample_rate = 2
        sampled_index = np.arange(0, N, sample_rate)
        h36m_motion_poses = pose_info[sampled_index]

        T = h36m_motion_poses.shape[0]
        h36m_motion_poses = h36m_motion_poses.reshape(T, 32, 3)
        return h36m_motion_poses

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.h36m_motion_input_length + self.h36m_motion_target_length)
        motion = self.h36m_seqs[idx][frame_indexes]

        h36m_motion_input = motion[:self.h36m_motion_input_length] / 1000.
        h36m_motion_target = motion[self.h36m_motion_input_length:] / 1000.

        h36m_motion_input = h36m_motion_input.float()
        h36m_motion_target = h36m_motion_target.float()
        # feature
        h36m_feature_target=self.h36m_files_feature[idx][frame_indexes[-1]]
        # GT
        # h36m_feature_target=self.h36m_22seqs[idx][frame_indexes[-1]].reshape(66)/1000
        # GT_residual
        # h36m_feature_target=(self.h36m_22seqs[idx][frame_indexes[-1]]-self.h36m_22seqs[idx][frame_indexes[-(self.h36m_motion_target_length+1)]]).reshape(66)/1000
        # h36m_feature_target.fill(1)
        return h36m_motion_input, h36m_motion_target, h36m_feature_target




import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config  import config
from model import siMLPe as Model
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import pickle
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegFileWriter

import torch
from torch.utils.data import DataLoader
device = torch.device(config.device)

def plot_3d_motion(save_path='tmp.gif', joints=None, title=' ', figsize=(5, 5), fps=20, radius=3, vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))
    kinematic_tree = [[0, 1, 2, 3], [0, 4, 5, 6, 7], [0, 8, 9], [0, 4, 8, 9], [9, 10, 11], [9, 12, 13, 14, 15, 16], [9, 17, 18, 19, 20, 21]]

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#61CEB9", "#34C1E2", "#80B79A", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#B75A39", "#FF6D00", "#DDB50E", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index):
        if index%10==0:
            print(f"processing frame {index}.")
        ax.clear()
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax._dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 7:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if index>=0 and (index+1) % 5 == 0:
            plt.savefig(f"{save_path[:-4]}_{index:03d}.png")

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']
# results_keys = ['#2', '#4', '#8', '#10']

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().to(device).unsqueeze(0)
idct_m = torch.tensor(idct_m).float().to(device).unsqueeze(0)

def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)
    with open('h36m_files_feature.pkl', 'rb') as f:
        feature = pickle.load(f)
    flist=[(0,0),(7,400),(77,400),(95,0),(128,1500)]
    for (motion_input, motion_target, feature_target) in pbar:
        for i in range(0,5):
            feature_target[i]=torch.from_numpy(feature[flist[i][0]][flist[i][1]])
            motion_input[i]=motion_input[0]
        motion_input = motion_input.to(device)
        feature_target=feature_target.to(device)  
        b,n,c,_ = motion_input.shape
        num_samples += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        step = config.motion.h36m_target_length_train
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.to(device))
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_,feature_target.to(device))
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            output = output.reshape(-1, 22*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            # motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        # modifed
        # motion_pred = torch.cat(outputs, axis=1)[:,:motion_target.shape[1]]
        motion_pred = torch.cat(outputs, axis=1)[:,:25]
        for i in range(0,128):
            joints = motion_pred[i].reshape(-1,22,3).cpu().numpy()
            gt_joints = motion_input[i].reshape(-1,22,3).cpu().numpy()
            joints=np.concatenate((gt_joints,joints),axis=0)
            print(f"frames:{joints.shape[0]}")
            print(joints.min(), joints.max())
            plot_3d_motion(save_path=f"output/tmp{i}.gif",joints=joints, fps=20, radius=1,gt_frames=range(0,50))

        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape1
        motion_gt = motion_target.clone()
        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]

        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader) :

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36)

    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')

    shuffle = True
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    print(test(config, model, dataloader))

