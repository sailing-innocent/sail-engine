import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from module.utils.video.av import concat_video

if __name__ == "__main__":
    vid_path_1 = "D:/logs/gaussian_eval_inno_reprod/nerf_blender_lego.mp4"
    vid_path_2 = "D:/logs/gaussian_eval_vanilla/nerf_blender_lego.mp4"
    output_path = "D:/logs/gaussian_eval_inno_reprod/concat.mp4"
    concat_video(vid_path_1, vid_path_2, output_path)