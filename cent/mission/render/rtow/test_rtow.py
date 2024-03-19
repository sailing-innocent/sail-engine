import pytest 
from mission.config.env import get_env_config
import os 

env_config = get_env_config()
result_dir = os.path.join(env_config.result_path, "demo_rtow")
exe_dir = os.path.abspath("../bin/release")

@pytest.mark.app 
def test_rtow_01():
    exe_name = "demo_rtow_01.write_image.exe"
    out_name = "fig_demo_rtow_01_write_image"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_02():
    exe_name = "demo_rtow_02.simple_ray.exe"
    out_name = "fig_demo_rtow_02_simple_ray"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_03():
    exe_name = "demo_rtow_03.hit_sphere.exe"
    out_name = "fig_demo_rtow_03_hit_sphere"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_04():
    exe_name = "demo_rtow_04.antialiasing.exe"
    out_name = "fig_demo_rtow_04_antialiasing"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_05():
    exe_name = "demo_rtow_05.sphere_normal.exe"
    out_name = "fig_demo_rtow_05_sphere_normal"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_06():
    exe_name = "demo_rtow_06.hittable_world.exe"
    out_name = "fig_demo_rtow_06_hittable_world"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_07():
    exe_name = "demo_rtow_07.matte.exe"
    out_name = "fig_demo_rtow_07_matte"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_08():
    exe_name = "demo_rtow_08.lambertian.exe"
    out_name = "fig_demo_rtow_08_lambertian"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_09():
    exe_name = "demo_rtow_09.materials.exe"
    out_name = "fig_demo_rtow_09_materials"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_10():
    exe_name = "demo_rtow_10.dielectrics.exe"
    out_name = "fig_demo_rtow_10_dielectrics"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.app
def test_rtow_11():
    exe_name = "demo_rtow_11.advanced_camera.exe"
    out_name = "fig_demo_rtow_11_advanced_camera"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.current
def test_rtow_12():
    exe_name = "demo_rtow_12.final_draw.exe"
    out_name = "fig_demo_rtow_12_final_draw"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)
