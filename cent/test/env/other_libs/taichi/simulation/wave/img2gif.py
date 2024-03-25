import os
import imageio.v2 as imageio

path = "./optimized/"
imgs = os.listdir(path)


gifname = "result.gif"
duration = 0.1

images = []
for img in imgs:
    img = path + img
    images.append(imageio.imread(img))

imageio.mimwrite(gifname, images, 'GIF', duration=duration)
