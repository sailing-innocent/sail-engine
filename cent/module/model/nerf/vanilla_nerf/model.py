from module.network.mlp.nerf import NeRF
from .helpers import get_embedder, render_rays, run_network, batchify_rays, render 
import torch 
import torch.nn as nn 

class VanillaNeRFModel(nn.Module):
    def __init__(self, chunk = 1024 * 32):
        super().__init__()
        self.embed_fn, self.input_ch = get_embedder(10)
        self.embeddirs_fn, self.inputdirs_ch = get_embedder(4)
        skips = [4]
        self.model = NeRF(
            D = 8,
            W = 256,
            input_ch = self.input_ch,
            input_ch_views = self.inputdirs_ch,
            skips = skips
        ).cuda()
        self.model_fine = NeRF(
            D=8,
            W=256,
            input_ch=self.input_ch,
            input_ch_views=self.inputdirs_ch,
            skips=skips
        ).cuda()
        self.network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,embed_fn=self.embed_fn,embeddirs_fn=self.embeddirs_fn,netchunk=1024 * 4)
        self.chunk = chunk

    def forward(self, rays_o, rays_d, near, far):
        rgb, _, _, _ = render(self.chunk, rays_o, rays_d, near, far, self.model, self.network_query_fn, self.model_fine)
        return rgb

    def load_ckpt(self, ckpt_file_path: str):
        ckpt = torch.load(ckpt_file_path)
        self.model.load_state_dict(ckpt['network_fn_state_dict'])
        self.model_fine.load_state_dict(ckpt['network_fine_state_dict'])


