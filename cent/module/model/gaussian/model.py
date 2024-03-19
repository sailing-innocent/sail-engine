import torch 

class Gaussians2D:
    def __init__(self, N):
        self.N = N
        self.means_2d = torch.zeros((N, 2), dtype=torch.float32).cuda()
        self.covs_2d = 0.0001 * torch.ones((N, 3), dtype=torch.float32).cuda()
        self.covs_2d[:, 1] = 0
        self.depth_features = torch.ones((N, 1), dtype=torch.float32).cuda()
        self.color_features = torch.ones((N, 3), dtype=torch.float32).cuda()
        self.opacity_features = 0.5 * torch.ones((N, 1), dtype=torch.float32).cuda()

    def requires_grad(self):
        self.means_2d.requires_grad = True
        self.covs_2d.requires_grad = True
        self.color_features.requires_grad = True
        self.opacity_features.requires_grad = True  

    def parameters(self):
        # return [self.means_2d, self.covs_2d, self.color_features, self.opacity_features]
        return [self.means_2d, self.color_features, self.opacity_features]
    
    @classmethod 
    def default(cls):
        return cls(1)
    
    @classmethod 
    def random(cls, N_per_layers, N_layers=1):
        gs = cls(N_per_layers * N_layers)
        gs.means_2d = 2 * torch.rand((N_per_layers * N_layers, 2), dtype=torch.float32).cuda() - 1
        for i in range(N_layers):
            gs.depth_features[i * N_per_layers: (i + 1) * N_per_layers, 0] = i / N_layers
            gs.color_features[i * N_per_layers: (i + 1) * N_per_layers, 0] = i / N_layers
            gs.color_features[i * N_per_layers: (i + 1) * N_per_layers, 1] = 1 - i / N_layers
        return gs

class Gaussians3D:
    def __init__(self, N):
        self.N = N
        self.means_3d = torch.zeros((N, 3), dtype=torch.float32).cuda()
        self.scales = torch.ones((N, 3), dtype=torch.float32).cuda()
        self.rotq = torch.zeros((N, 4), dtype=torch.float32).cuda()
        self.color_features = torch.ones((N, 3), dtype=torch.float32).cuda()
        self.opacity_features = 0.5 * torch.ones((N, 1), dtype=torch.float32).cuda()