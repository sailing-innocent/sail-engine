import taichi as ti 
import taichi.math as tm

@ti.func 
def integrate(ray_origin, ray_direction, obj: ti.template()):
    flag, inside, isect = ray_sphere_intersect(ray_origin, ray_direction, obj)
    background_color = tm.vec3(0.572, 0.772, 0.921)
    color = background_color
    # ray marcher
    if flag:
        step_size = 0.2
        absorption = 0.1
        scattering = 0.1
        density = 5
        ns = tm.ceil((isect.t1 - isect.t0) / step_size)
        step_size = (isect.t1 - isect.t0) / ns 

        light_dir = tm.vec3(0.0, 1.0, 0.0)
        light_color = tm.vec3(1.3, 0.3, 0.9)

        transparency = 1.0

        result = tm.vec3(0.0, 0.0, 0.0)

        g = 0.8 # asymmetry factor of the phase function
        cos_theta = tm.dot(ray_direction, light_dir)

        backward_marching = True

        if backward_marching:
            for i in range(ns):
                t = isect.t1 - step_size * (i + 0.5)
                p = ray_origin + t * ray_direction

                sample_transparency = tm.exp(-step_size * (scattering + absorption + 0.4))
                transparency *= sample_transparency

                # add light
                flag, vinside, vsect = ray_sphere_intersect(p, light_dir, obj)
                if flag and vinside:
                    light_attenuation = tm.exp(-density * vsect.t1 * (scattering + absorption))
                    result += transparency * light_attenuation * light_color * scattering * density * step_size # * phase(g, cos_theta) 
                result *= sample_transparency
        else:
            # forward
            for i in range(ns):
                t = isect.t0 + step_size * (i + 0.5)
                p = ray_origin + t * ray_direction

                sample_attenuation = tm.exp(-step_size * (scattering + absorption))
                transparency *= sample_attenuation

                # add light
                flag, vinside, vsect = ray_sphere_intersect(p, light_dir, obj)
                if flag:
                    light_attenuation = tm.exp(-density * vsect.t1 * (scattering + absorption))
                    result += light_attenuation * light_color * phase(g, cos_theta) * scattering * density * step_size
        
        color = background_color * transparency + result
        # color = tm.vec3(1.0, 0.0, 0.0)
    # background
    return color
@ti.func
def phase(g: float, cos_theta: float):
    denom = 1 + g * g - 2 * g * cos_theta 
    return 1 / (4 * tm.pi) * (1 - g * g) / (denom * tm.sqrt(denom))

@ti.func
def phase_uniform():
    return 1 / (4 * tm.pi)

@ti.func
def ray_sphere_intersect(ro, rd, s):
    # ray-sphere intersection
    a = tm.dot(rd, rd)
    b = 2 * tm.dot(rd, ro - s.center)
    c = tm.dot(ro - s.center, ro - s.center) - s.radius * s.radius
    flag, x1, x2 = solve_quadratic(a, b, c)
    inside = False 
    isect = ISectData(
        x1, x2, tm.vec3(0, 0, 0), tm.vec3(0, 0, 0))
    if (flag):
        if (isect.t0 < 0):
            if (isect.t1 < 0):
                flag = False 
            else:
                inside = True
                isect.t0 = 0

    return flag, inside, isect

@ti.func
def solve_quadratic(a, b, c):
    d = b * b - 4 * a * c
    flag = False 
    x1 = 0.0
    x2 = 0.0
    if d < 0:
        pass 
    elif d > 0:
        d = ti.sqrt(d)
        x1 = (-b - d) / (2 * a)
        x2 = (-b + d) / (2 * a)
        flag = True
    else:
        x1 = x2 = -b / (2 * a)
        flag = True

    return flag, x1, x2        

@ti.dataclass
class Sphere:
    center: tm.vec3
    radius: float

@ti.dataclass
class ISectData:
    t0: float
    t1: float
    pHit: tm.vec3
    nHit: tm.vec3

@ti.kernel
def ray_marcher(pixels: ti.template(), n: int):
    s = Sphere(tm.vec3(0, 0, 0), 1)
    for i, j in pixels:
        # ray origin
        ro = tm.vec3(0, 0, 3)
        # ray direction
        rd = tm.normalize(tm.vec3(i / n - 0.5, j / n - 0.5, -1))
        pixels[i, j] = integrate(ro, rd, s)

class TaichiRayMarcher:
    def __init__(self):
        self.n = 512
        ti.init(arch=ti.cuda)
        self.gui = ti.GUI("Ray Marcher", res=(self.n, self.n))
        self.pixels = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.n, self.n))

    def run(self):
        while self.gui.running:
            ray_marcher(self.pixels, self.n)
            self.gui.set_image(self.pixels)
            self.gui.show()
