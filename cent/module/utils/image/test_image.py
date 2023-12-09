import pytest 
from module.utils.image.basic import Image

@pytest.mark.current 
def test_image():
    img = Image()
    assert img.W == 800
    assert img.H == 600
    assert img.C == 3
    assert img.shape == (600, 800, 3)
    img.load_from_file('doc/figure/asset/logo_60x60.png')
    assert img.W == 60
    assert img.H == 60
    assert img.C == 4
    img.show(immediate=True)
    img.save("test.png")

