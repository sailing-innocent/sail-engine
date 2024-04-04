import torch 

from typing import NamedTuple
class Gaussians2DTrainArgs(NamedTuple):
    lr_means_2d = 1e-2
    lr_covs_2d = 1e-9
    lr_color = 1e-2
    lr_opacity = 1e-9

class Gaussians2D:
    def __init__(self):
        self.means_2d = torch.empty(0)
        self.covs_2d = torch.empty(0)
        self.depth_features = torch.empty(0)
        self.color_features = torch.empty(0)
        self.opacity_features = torch.empty(0)
        self.optim = None 

    def _create(self, N):
        self.N = N 
        self.means_2d = torch.zeros((N, 2), dtype=torch.float32).cuda()
        self.covs_2d = 0.0001 * torch.ones((N, 3), dtype=torch.float32).cuda()
        self.covs_2d[:, 1] = 0
        self.depth_features = torch.ones((N, 1), dtype=torch.float32).cuda()
        self.color_features = torch.ones((N, 3), dtype=torch.float32).cuda()
        self.opacity_features = 0.5 * torch.ones((N, 1), dtype=torch.float32).cuda()

    def training_setup(self, training_args: Gaussians2DTrainArgs):
        l = [
            {'params': [self.means_2d], 'lr': training_args.lr_means_2d, "name": "means_2d"},
            {'params': [self.covs_2d], 'lr': training_args.lr_covs_2d, "name": "covs_2d"},
            {'params': [self.color_features], 'lr': training_args.lr_color, "name": "color_features"},
            {'params': [self.opacity_features], 'lr': training_args.lr_opacity, "name": "opacity"}
        ]
        self.optim = torch.optim.AdamW(l, lr=0.0, eps=1e-15)

    def requires_grad(self):
        self.means_2d.requires_grad = True
        self.covs_2d.requires_grad = True
        self.color_features.requires_grad = True
        self.opacity_features.requires_grad = True  

    def clone_detach(self):
        self.means_2d = self.means_2d.detach().clone()
        self.covs_2d = self.covs_2d.detach().clone()
        self.color_features = self.color_features.detach().clone()
        self.opacity_features = self.opacity_features.detach().clone()
    
    def parameters(self):
        return [self.means_2d, self.covs_2d, self.color_features, self.opacity_features]
        # return [self.means_2d, self.color_features, self.opacity_features]
        # return [self.means_2d, self.color_features]
    
    @classmethod 
    def default(cls):
        gs = cls()
        gs._create(1)
        return gs
    
    @classmethod 
    def random(cls, N_per_layers, N_layers=1):
        gs = cls()
        gs._create(N_per_layers * N_layers)
        gs.means_2d = 2 * torch.rand((N_per_layers * N_layers, 2), dtype=torch.float32).cuda() - 1
        for i in range(N_layers):
            gs.depth_features[i * N_per_layers: (i + 1) * N_per_layers, 0] = i / N_layers
            gs.color_features[i * N_per_layers: (i + 1) * N_per_layers, 0] = i / N_layers
            gs.color_features[i * N_per_layers: (i + 1) * N_per_layers, 1] = 1 - i / N_layers
        return gs

    def subview(self, from_idx, to_idx):
        gs = Gaussians2D(to_idx - from_idx)
        gs.means_2d = self.means_2d[from_idx:to_idx]
        gs.covs_2d = self.covs_2d[from_idx:to_idx]
        gs.depth_features = self.depth_features[from_idx:to_idx]
        gs.color_features = self.color_features[from_idx:to_idx]
        gs.opacity_features = self.opacity_features[from_idx:to_idx]
        return gs
