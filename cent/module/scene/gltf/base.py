from abc import ABC, abstractmethod 

class GLTFObject(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def from_gltf(self, gltf_json):
        pass 

    @abstractmethod
    def to_gltf(self, output_dir: str = ""):
        pass
    