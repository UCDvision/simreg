import torch
import torch.nn as nn

import models.resnet as resnet
from models.mobilenet import MobileNetV2 as mobilenet
from models.resnet_byol import resnet50 as byol_resnet50
from mlp_arch import get_mlp, get_linear_proj
from tools import get_shuffle_ids


class SimReg(nn.Module):
    def __init__(self, args, arch_t, arch_s, teach_wts):
        super(SimReg, self).__init__()

        # save parameters
        self.args = args

        # Create encoders and projection layers
        # Both teacher and student encoders must have the same arch
        if 'resnet' in arch_s:
            self.encoder_q = resnet.__dict__[arch_s]()
        elif 'mobilenet' in arch_s:
            self.encoder_q = mobilenet()

        if 'byol_resnet50' in arch_t:
            self.encoder_t = byol_resnet50()
        elif 'sup_resnet50' in arch_t and not args.use_cache:
            self.encoder_t = resnet.__dict__['resnet50'](fc_dim=1000)
        elif 'resnet' in arch_t and not args.use_cache:
            self.encoder_t = resnet.__dict__[arch_t]()

        # save output embedding dimensions
        if not args.use_cache:
            if args.teacher_fc:
                feat_dim_t = self.encoder_t.fc.out_features
            else:
                feat_dim_t = self.encoder_t.fc.in_features
        else:
            teach_dims = {'resnet50': 2048, 'resnet50x4': 8192}
            feat_dim_t = teach_dims[arch_t]

        feat_dim_q = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Sequential()

        self.feat_dim_t = feat_dim_t
        if self.args.n_mlp_layers == 2:
            hidden_dims = [feat_dim_q, feat_dim_q*2]
        elif self.args.n_mlp_layers == 4:
            hidden_dims = [feat_dim_q, feat_dim_q*2, feat_dim_q, feat_dim_q*2]
        else:
            sys.exit('n_mlp_layers = %d is not implemented!!!' % self.args.n_mlp_layers)

        if not args.use_cache and not args.teacher_fc:
            self.encoder_t.fc = nn.Sequential()

        # prediction layer
        if args.linear_pred:
            self.predict_q = get_linear_proj(proj_dim, feat_dim_t)
        else:
            self.predict_q = get_mlp(hidden_dims, feat_dim_t, self.args.n_mlp_layers)

        if not args.use_cache:
            # Load teacher parameters from pretrained model
            ckpt = torch.load(teach_wts)
            # Modify the names of checkpoint keys for compatibility with current model names.
            # Stored weight names have the form 'model.encoder_q.<param_name>'. This is modified to just '<param_name>'.
            checkpoint = {}
            if 'state_dict' in ckpt:
                sd = ckpt['state_dict']
            else:
                sd = ckpt
            for key, val in sd.items():
                # new_key = '.'.join(key.split('.')[2:])
                new_key = key.replace('module.encoder_q.', '')
                checkpoint[new_key] = val
            msg = self.encoder_t.load_state_dict(checkpoint, strict=False)
            print(msg)
            assert (len(msg.missing_keys) == 0), 'Missing keys in teacher weights!'
        elif args.use_cache:
            # Load teacher features from cache
            self.t_feats = torch.load(self.args.teacher_feats)

        # Disable gradients for teacher network
        if not args.use_cache:
            for param in self.encoder_t.parameters():
                param.requires_grad = False
            self.encoder_t.eval()

    @torch.no_grad()
    def data_parallel(self):
        self.encoder_q = torch.nn.DataParallel(self.encoder_q)
        if not self.args.use_cache:
            self.encoder_t = torch.nn.DataParallel(self.encoder_t)
        self.predict_q = torch.nn.DataParallel(self.predict_q)

    def forward(self, im_q, im_t, names=None):
        # compute query features from encoder and prediction MLP
        feat_q = self.encoder_q(im_q)
        query = self.predict_q(feat_q)
        query = nn.functional.normalize(query, dim=1)

        # compute target features
        with torch.no_grad():
            # shuffle targets
            shuffle_ids, reverse_ids = get_shuffle_ids(im_t.shape[0])
            im_t = im_t[shuffle_ids]

            # forward through the target encoder
            if not self.args.use_cache:
                current_target = self.encoder_t(im_t)
            else:
                current_target = torch.zeros(feat_q.shape[0], self.feat_dim_t)
                for i, name in enumerate(names):
                    current_target[i] = self.t_feats[name]
                current_target = current_target.cuda()
            current_target = nn.functional.normalize(current_target, dim=1)

            # undo shuffle
            if not self.args.use_cache:
                current_target = current_target[reverse_ids].detach()

        # calculate regression loss
        if self.args.mse_nonorm:
            # (bs, n_dim), (bs, n_dim) --> (bs)
            # mse using un-normalized vectors
            dist = ((query - current_target) ** 2).sum(-1)
        else:
            # mse using normalized vectors = 2 - 2 * cosine dist
            dist = 2 - 2 * torch.einsum('bc,bc->b', [query, current_target])
        loss = dist.mean()
        return loss
