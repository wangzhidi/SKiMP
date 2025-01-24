import argparse
import os, sys
import json
import math
import numpy as np
import copy
from einops import repeat


from config import config
from model import siMLPe as Model
from datasets.h36m import H36MDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from datasets.h36m_eval import H36MEval


from test import test

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device(config.device)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

args = parser.parse_args()

torch.use_deterministic_algorithms(True)
acc_log = open(args.exp_name, 'a')
torch.manual_seed(args.seed)
writer = SummaryWriter()

config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))

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

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(h36m_motion_input, h36m_motion_target, h36m_feature_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :
    motion_feats = h36m_motion_input.clone()
    # # early_concatenate
    # length = motion_feats.shape[-2]
    # # 66
    # h36m_feature_target = repeat(h36m_feature_target, 'B C -> B N C', N=length)
    # # 3300
    # # sketch_feature = sketch_feature.reshape(sketch_feature.shape[0],50,66)
    # motion_feats = torch.cat([motion_feats, h36m_feature_target], dim=-1)



    if config.deriv_input:
        b,n,c = motion_feats.shape
        h36m_motion_input_ = motion_feats.clone()
        h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.to(device))
    else:
        h36m_motion_input_ = motion_feats.clone()
        
    h36m_feature_target=h36m_feature_target.clone()
    motion_pred = model(h36m_motion_input_.to(device),h36m_feature_target.to(device))
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    if config.deriv_output:
        offset = h36m_motion_input[:, -1:].to(device)
        motion_pred = motion_pred[:, :config.motion.h36m_target_length] + offset
    else:
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]

    # b,n,c = h36m_motion_target.shape
    # motion_pred = motion_pred.reshape(b,n,22,3).reshape(-1,3)
    # h36m_motion_target = h36m_motion_target.to(device).reshape(b,n,22,3).reshape(-1,3)
    # loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))
    

    b, n, c = h36m_motion_target.shape

    # Reshape predictions and targets
    motion_pred = motion_pred.reshape(b, n, 22, 3).reshape(-1, 3)
    h36m_motion_target = h36m_motion_target.to(device).reshape(b, n, 22, 3).reshape(-1, 3)

    # Create a weight vector (larger weight for earlier frames)
    weights = torch.linspace(1, 0.1, n).to(device)  # Linearly decreasing weight from 1 to 0.1
    weights = weights.view(1, n, 1).repeat(b, 1, 22)  # Repeat for each batch and joint (22 joints)

    # Compute the norm between prediction and target
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1) * weights.view(-1))



    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,22,3)
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b,n,22,3)
        dmotion_gt = gen_velocity(motion_gt)
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss = loss + dloss
    else:
        loss = loss.mean()

    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    writer.add_scalar('LR/train', current_lr, nb_iter)

    return loss.item(), optimizer, current_lr

model = Model(config)
model.train()
model.to(device)

config.motion.h36m_target_length = config.motion.h36m_target_length_train
dataset = H36MDataset(config, 'train', config.data_aug)
import pickle
with open('h36m_files.pkl', 'wb') as f:
    pickle.dump(dataset._h36m_files, f)
with open('h36m_seqs.pkl', 'wb') as f:
    pickle.dump(dataset.h36m_seqs, f)
with open('h36m_names.pkl', 'wb') as f:
    pickle.dump(dataset._file_names, f)


shuffle = True
sampler = None
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

eval_config = copy.deepcopy(config)
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
eval_dataset = H36MEval(eval_config, 'test')

shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                        num_workers=1, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)
with open('h36m_eval_seqs.pkl', 'wb') as f:
    pickle.dump(eval_dataset.h36m_22seqs, f)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)
  
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

while (nb_iter + 1) < config.cos_lr_total_iters:

    for (h36m_motion_input, h36m_motion_target, h36m_feature_target) in dataloader:

        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, h36m_feature_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        avg_loss += loss
        avg_lr += current_lr

        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every

            print_and_log_info(logger, "Iter {} Summarn  y: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % config.save_every ==  0 :
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            model.eval()
            acc_tmp = test(eval_config, model, eval_dataloader)
            print(acc_tmp)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            model.train()

        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()
