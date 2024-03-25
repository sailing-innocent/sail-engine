import torch 
import torch.nn as nn 

background_color = torch.Tensor([0.572, 0.772, 0.921])
class ISectData:
    def __init__(self, 
        t0 = 0.0, 
        t1 = 1.0, 
        pHit = torch.Tensor([0,0,0]), 
        nHit = torch.Tensor([0,0,0])):
        self.t0 = t0
        self.t1 = t1
        self.pHit = pHit
        self.nHit = nHit
class Sphere:
    def __init__(self, center=torch.Tensor([0,0,0]),radius=1):
        super().__init__()
        self.center = center 
        self.radius = radius

def solve_quadratic(a, b, c):
    d = b * b - 4 * a * c
    flag = False
    x1 = 0.0
    x2 = 0.0
    if d < 0:
        return flag, x1, x2
    elif d > 0:
        d = torch.sqrt(d)
        x1 = (-b - d) / (2 * a)
        x2 = (-b + d) / (2 * a)
        flag = True
    else:
        x1 = x2 = -b / (2 * a)
        flag = True
    
    return flag, x1, x2

def ray_sphere_intersect(ro, rd, s):
    a = torch.dot(rd, rd)
    b = 2 * torch.dot(rd, ro - s.center)
    c = torch.dot(ro - s.center, ro - s.center) - s.radius * s.radius
    flag, x1, x2 = solve_quadratic(a, b, c)
    inside = False
    isect = ISectData(x1, x2)
    if (flag):
        if (isect.t0 < 0):
            if (isect.t1 < 0):
                flag = False
            else:
                inside = True
                isect.t0 = 0

    return flag, inside, isect

def integrate(ray_origin, ray_direction, obj):
    flag, inside, isect = ray_sphere_intersect(ray_origin, ray_direction, obj)
    color = background_color
    # ray marcher
    if flag:
        step_size = 0.2
        absorption = 0.1
        scattering = 0.1
        density = 6.0
        ns = torch.ceil((isect.t1 - isect.t0) / step_size)
        step_size = (isect.t1 - isect.t0) / ns 

        light_dir = torch.Tensor([0.0, 1.0, 0.0])
        light_color = torch.Tensor([1.3, 0.3, 0.9])

        transparency = 1.0

        result = torch.Tensor([0.0, 0.0, 0.0])
        for i in range(int(ns)):
            t = isect.t1 - (i + 0.5) * step_size
            p = ray_origin + t * ray_direction

            sample_transparency = torch.exp(-step_size * (scattering + absorption + 0.4))
            transparency *= sample_transparency

            flag, vinside, vsect = ray_sphere_intersect(p, light_dir, obj)
            if flag:
                light_attenuation = torch.exp(-density * vsect.t1 * (scattering + absorption))
                result += light_attenuation * light_color * transparency * scattering * density * step_size
            result *= sample_transparency
        
        color = background_color * transparency + result

    # background
    return color

class RayMarcherModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scene: torch.Tensor, dirs: torch.Tensor):
        assert scene.shape[1] == 4
        assert scene.shape[0] == 1
        sphere = Sphere(scene[0,0:3], scene[0,3])
        W = dirs.shape[0]
        H = dirs.shape[1]
        pixels = torch.ones_like(dirs)
        for i in range(W):
            for j in range(H):
                ro = torch.Tensor([0,0,3])
                rd = dirs[i,j,:]
                pixels[i,j,:] = integrate(ro, rd, sphere)
        return pixels

