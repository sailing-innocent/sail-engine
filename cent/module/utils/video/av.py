import numpy as np 
import av 
import os 

# read img of color in [0, 1] float
def write_mp4(
    img_list: list, 
    video_name: str, 
    target_path: str, 
    fps: int = 24):
    total_frames = len(img_list)
    fname = os.path.join(target_path, video_name + ".mp4")
    container = av.open(fname, mode="w")
    print("writing video to {}".format(fname))
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = img_list[0].shape[0]
    stream.height = img_list[0].shape[1]
    stream.pix_fmt= "yuv420p"
    for frame_i in range(total_frames):
        img = img_list[frame_i]
        img = np.round(255 * img).astype(np.uint8)
        img = np.clip(img, 0, 255)

        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush stream
    for packet in stream.encode():
        container.mux(packet)

    # Close the file
    container.close()
    return True

import av.datasets

def concat_video(v1_path, v2_path, output_path):
    container_1 = av.open(v1_path)
    container_2 = av.open(v2_path)
    container_o = av.open(output_path, mode="w")
    stream = container_o.add_stream("mpeg4")
    width = container_1.streams.video[0].width
    height = container_1.streams.video[0].height

    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    N_frames = container_1.streams.video[0].frames

    for i in range(N_frames):
        frame_1 = next(container_1.decode(video=0))
        frame_2 = next(container_2.decode(video=0))

        rate = i / N_frames
        rw = int(rate * width)
        # concat left | right
        img1 = np.array(frame_1.to_image())
        img2 = np.array(frame_2.to_image())
        img1_view = img1[:, :rw, :]
        img2_view = img2[:, rw:, :]
        # concat
        img = np.concatenate([img1_view, img2_view], axis=1)
        # add concat line
        if (rw < width - 2) and (rw > 2):
            img[:, rw-2:rw+2, :] = 0
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container_o.mux(packet)

    # close
    container_o.close()
    container_1.close()
    container_2.close()
    return True 