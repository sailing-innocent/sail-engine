import pytest 
from mission.config.env import get_env_config
import os 

env_config = get_env_config()
result_dir = os.path.join(env_config.result_path, "rtow")
exe_dir = os.path.abspath("../bin/release")

@pytest.mark.app 
def test_rtow_01():
    exe_name = "demo_rtow_01.write_image.exe"
    out_name = "fig_demo_rtow_01_write_image"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)

@pytest.mark.current 
def test_rtow_02():
    exe_name = "demo_rtow_02.simple_ray.exe"
    out_name = "fig_demo_rtow_02_simple_ray"
    exe_path = os.path.join(exe_dir, exe_name)
    cmd = f"{exe_path} {result_dir} {out_name}"
    os.system(cmd)