from abc import ABC, abstractmethod

class BlenderSceneObject(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def from_blender(self, scene):
        pass

    