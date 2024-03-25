import json 
import os 

from .base import GLTFObject
from .mesh import Mesh 
from .data import Buffer, BufferView, Accessor, Image
from .material import Material
from .texture import Texture, Sampler 

class Scene(GLTFObject):
    def __init__(self):
        self._scene_graph = SceneGraph()
        self._nodes = []
        self._meshes = []
        self._materials = []
        self._samplers = []
        self._images = []
        self._textures = []
        self._buffers = []
        self._buffer_views = []
        self._accessors = []

        self.gltf_obj_map = {
            "nodes": SceneNode,
            "meshes": Mesh,
            "materials": Material,
            "samplers": Sampler,
            "textures": Texture,
            "images": Image,
            "buffers": Buffer,
            "bufferViews": BufferView,
            "accessors": Accessor
        }

    def try_from_gltf(self, gltf_json, name: str):
        obj_list = []
        if name in self.gltf_obj_map and gltf_json[name]:
            for obj in gltf_json[name]:
                obj_inst = self.gltf_obj_map[name]()
                obj_inst.from_gltf(obj)
                obj_list.append(obj_inst)
        
        return obj_list


    def from_gltf(self, gltf_json):
        self._scene_graph.from_gltf(gltf_json["scenes"][0])
        self._nodes = self.try_from_gltf(gltf_json, "nodes")
        self._meshes = self.try_from_gltf(gltf_json, "meshes")
        # self._materials = self.try_from_gltf(gltf_json, "materials")
        # self._samplers = self.try_from_gltf(gltf_json, "samplers")
        # self._images = self.try_from_gltf(gltf_json, "images")
        # self._textures = self.try_from_gltf(gltf_json, "textures")
        self._buffers = self.try_from_gltf(gltf_json, "buffers")
        self._buffer_views = self.try_from_gltf(gltf_json, "bufferViews")
        self._accessors = self.try_from_gltf(gltf_json, "accessors")

    def to_gltf(self, output_dir: str = ""):
        obj = {}
        obj["scenes"] = [self._scene_graph.to_gltf()]
        obj["nodes"] = [node.to_gltf() for node in self._nodes]
        obj["meshes"] = [mesh.to_gltf() for mesh in self._meshes]
        obj["buffers"] = []
        for buf in self._buffers:
            buf_json, buf_data = buf.to_gltf()
            obj["buffers"].append(buf_json)
            with open(os.path.join(output_dir, buf_json["uri"]), "wb") as f:
                f.write(buf_data)

        obj["bufferViews"] = [buffer_view.to_gltf() for buffer_view in self._buffer_views]
        obj["accessors"] = [accessor.to_gltf() for accessor in self._accessors]
        obj["materials"] = [material.to_gltf() for material in self._materials]

        obj["asset"] = {
            "version": "2.0"
        }
        return obj

class SceneGraph(GLTFObject):
    def __init__(self):
        self._nodes = []

    def add_node(self, node):
        self._nodes.append(node)

    def from_gltf(self, gltf_json):
        self._nodes = gltf_json["nodes"]

    def to_gltf(self):
        obj = {}
        obj["nodes"] = self._nodes 
        return obj

class SceneNode:
    def __init__(self):
        self._children = []
        self._mesh = None 
        self._camera = None 
        self._matrix = []
        self._type = "Matrix"
        self._scale = []
        self._rotation = []
        self._translation = []
        self._extras = {
            "opaque": True,
            "materials": []
        }
    
    def add_child(self, child_node):
        self._children.append(child_node)

    def from_gltf(self, gltf_json):
        if "matrix" in gltf_json:
            self._matrix = gltf_json["matrix"]
            self._type = "Matrix"
        if "scale" in gltf_json:
            self._translation = gltf_json["translation"]
            self._scale = gltf_json["scale"]
            self._rotation = gltf_json["rotation"]
            self._type = "SRT"   
        if "mesh" in gltf_json:
            self._mesh = gltf_json["mesh"]

    def to_gltf(self):
        obj = {}
        obj["matrix"] = self._matrix
        if self._mesh is not None:
            obj["mesh"] = self._mesh
    
        obj["extras"] = self._extras
        return obj

