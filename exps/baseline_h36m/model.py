import copy

import torch
from torch import nn
from mlp import build_mlps
from einops.layers.torch import Rearrange
from einops import repeat

class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        seq = self.config.motion_mlp.seq_len
        self.arr0 = Rearrange('b n d -> b d n')
        self.arr1 = Rearrange('b d n -> b n d')

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            # self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
            # concatenate
            self.motion_fc_in = nn.Linear(2*self.config.motion.dim, self.config.motion.dim)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct)
        else:
            # concatenate
            # self.motion_fc_out = nn.Linear(2 * self.config.motion.dim, self.config.motion.dim)
            
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input, sketch_feature):
        # sketch feature [256, 1, 66]
        # [256,50,66]
        motion_feats = motion_input
        # # concatenate
        length = motion_feats.shape[-2]
        # # 66
        sketch_feature = repeat(sketch_feature, 'B C -> B N C', N=length)
        # # 3300
        # sketch_feature = sketch_feature.reshape(sketch_feature.shape[0],50,66)
        motion_feats = torch.cat([motion_feats, sketch_feature], dim=-1)

        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_feats)  #[256,50,66]
            motion_feats = self.arr0(motion_feats) #[256,66,50]

        # replace
        # motion_feats[:,:,0] = sketch_feature[:, :]
        # add-3300
        # sketch_feature=sketch_feature.reshape(-1,66,50)*100
        # motion_feats = motion_feats + sketch_feature
        # add
        # motion_feats = motion_feats + sketch_feature[:, :, None]

        motion_feats = self.motion_mlp(motion_feats)    #[256,66,50]





        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)#[256,50,66]

        return motion_feats

