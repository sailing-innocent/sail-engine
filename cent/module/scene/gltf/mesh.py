import json 
from .base import GLTFObject

class MeshPrimitive(GLTFObject):
    def __init__(self):
        self._attributes = VertexAttributes()
        self._indices = None 
        self._material = None 
        self._mode = 4

    def from_gltf(self, gltf_json):
        self._attributes.from_gltf(gltf_json["attributes"])
        self._indices = gltf_json["indices"]
        self._material = gltf_json["material"]

    def to_gltf(self):
        obj = {}
        obj["attributes"] = self._attributes.to_gltf()
        obj["indices"] = self._indices
        # obj["material"] = self._material
        # obj["mode"] = self._mode
        return obj


class VertexAttributes(GLTFObject):
    def __init__(self, has_normal = True, has_tangent = True, n_tex_coord = 2):
        self._has_normal = has_normal
        self._has_tangent = has_tangent
        self._n_tex_coord = n_tex_coord
        self._POSITION = 0
        self._NORMAL = 1
        self._TANGENT = 2
        self._TEXCOORD_LIST = []

    def from_gltf(self, gltf_json):
        self._has_normal = "NORMAL" in gltf_json
        self._has_tangent = "TANGENT" in gltf_json
        self._POSITION = gltf_json["POSITION"]
        offset = 1
        if self._has_normal:
            self._NORMAL = gltf_json["NORMAL"]
            offset += 1
        if self._has_tangent:
            self._TANGENT = gltf_json["TANGENT"]
            offset += 1

        self._n_tex_coord = len(gltf_json.keys()) - offset 
        for i in range(self._n_tex_coord):
            self._TEXCOORD_LIST.append(gltf_json["TEXCOORD_" + str(i)])

    def to_gltf(self):
        obj = {}
        obj["POSITION"] = self._POSITION
        if self._has_normal:
            obj["NORMAL"] = self._NORMAL
        if self._has_tangent:
            obj["TANGENT"] = self._TANGENT
        for i in range(self._n_tex_coord):
            obj["TEXCOORD_" + str(i)] = self._TEXCOORD_LIST[i]
        return obj

    def offset_all(self, i: int):
        self._POSITION += i
        if self._has_normal:
            self._NORMAL += i
        if self._has_tangent:
            self._TANGENT += i
        for i in range(self._n_tex_coord):
            self._TEXCOORD_LIST[i] += i


class Mesh(GLTFObject):
    def __init__(self):
        self._primitives = []
        self._extras = {}

    def from_gltf(self, gltf_json):
        for p in gltf_json["primitives"]:
            inst = MeshPrimitive()
            inst.from_gltf(p)
            self._primitives.append(inst)

    def to_gltf(self):
        obj = {}
        obj["primitives"] = [p.to_gltf() for p in self._primitives]
        obj["extras"] = self._extras
        return obj
