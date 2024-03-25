from abc import ABC, abstractmethod

class SceneObject(ABC):
    def __init__(self):
        pass 

    # @abstractmethod 
    # def from_gltf(self, gltf_json):
    #     pass 

    @abstractmethod
    def to_gltf(self):
        pass 

    @abstractmethod
    def from_blender(self, blender_scene):
        pass 
    