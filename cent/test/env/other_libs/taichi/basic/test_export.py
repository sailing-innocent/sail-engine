import pytest 
import taichi as ti 

@pytest.mark.app
def test_export_video():
    ti.init()

    pixels = ti.field(ti.u8, shape=(512,512,3))

    @ti.kernel
    def paint():
        for i, j, k in pixels:
            pixels[i, j, k] = ti.random() * 255 # random noise

    result_dir = "./output"
    video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    for i in range(50):
        paint()

        pixels_img = pixels.to_numpy()
        video_manager.write_frame(pixels_img)
        print(f'\rFrame {i+1}/50 frame is recorded', end='')

    print()
    print('Exporting gif file')
    video_manager.make_video(gif=False, mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
