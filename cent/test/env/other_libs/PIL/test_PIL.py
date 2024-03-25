from PIL import Image 
import numpy as np 
SIZE = 512


def test_grayscale():
    image = Image.new("L", (SIZE, SIZE))
    image.save("output/test_PIL_grayscale.png")
    assert 0 == 0

def test_from_array():
    data = np.ones((SIZE, SIZE), dtype=float) * 255.0
    img = Image.fromarray(data).convert("L")
    img.save("output/test_PIL_array_white.png")
    assert 0 == 0