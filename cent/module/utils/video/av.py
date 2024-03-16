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