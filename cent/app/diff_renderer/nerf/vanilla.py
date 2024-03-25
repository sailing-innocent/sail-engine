import torch 

class NeRFRenderer:
    def __init__(self):
        pass 

    def render(self, camera, model, near, far):
        # generate rays from camera
        rays_o, rays_d = camera.rays
        rays_o = torch.from_numpy(rays_o).float().cuda()
        rays_d = torch.from_numpy(rays_d).float().cuda()
        rgb = model(rays_o, rays_d, near, far)
        return rgb

    def render_rays(self, rays_o, rays_d, model, near, far):
        rgb = model(rays_o, rays_d, near, far)
        return rgb

def create_nerf_renderer():
    return NeRFRenderer()