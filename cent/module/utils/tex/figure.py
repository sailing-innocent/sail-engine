from utils.image.basic import Image 

class ImageFigure:
    def __init__(self, img: Image):
        self.img = img 

    def export(self, filepath=None):
        pass 
        # if filepath exists, write to file path
        # else, return the content ( name, build_str )

class FigPack:
    def __init__(self):
        self.figs = []
        self.name = "fig_pack_dummy"

    def export(self, filepath=None):
        pass 
        # if filepath exists, write to file path
        # else, return the content ( name, build_str )