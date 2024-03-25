from .base import GLTFObject

class Material(GLTFObject):
    def __init__(self):
        pass 
    
    def from_gltf(self, gltf_json):
        pass

    def to_gltf(self):
        obj = {}
        return obj