from .base import BlenderSceneObject
import json 
import bmesh 
import os 
import struct

class Mesh(BlenderSceneObject):
    def __init__(self):
        self.use_normal = False 
        self.use_tangent = False 
        self.n_tex_coord = 0

        self.vert_pos = []
        # TODO fetch normal, tangent and texcoord
        self.vert_normal = []
        self.vert_tangent = []
        self.indices = []
        self.name = "" # Single Identifier

    def from_blender(self, mesh):
        self.name = mesh.name
        bm = bmesh.new()
        bm.from_mesh(mesh.data)
        # place holder for float4
        self.vert_pos = [[v.co[0], v.co[1], v.co[2], 0.0] for v in bm.verts]
        # flatten
        self.vert_pos = [i for v in self.vert_pos for i in v]
        # print(self.vert_pos)
        num_triangles = len(bm.faces) * 2
        self.indices = [[[
            f.verts[0].index, 
            f.verts[1].index,
            f.verts[2].index],[
            f.verts[0].index,
            f.verts[2].index,
            f.verts[3].index]]
        for f in bm.faces]
        self.indices = [i for f in self.indices for i3 in f for i in i3]
        # print(self.indices)
        # transform

class SceneNode(BlenderSceneObject):
    def __init__(self):
        self._type = "MESH"
        self._mesh = None 
        self._matrix = None 

    def from_blender(self, obj):
        self._type = obj.type 
        self._matrix = [i for v in obj.matrix_world for i in v]
        if obj.type == 'MESH':
            self._mesh = Mesh()
            self._mesh.from_blender(obj)
        elif obj.type == 'CAMERA':
            return
        elif obj.type == 'LIGHT':
            return
        else:
            print("get unknown")
            print(obj.name)
    

class Scene(BlenderSceneObject):
    def __init__(self):
        self._nodes = []
        self._meshes = {}

    def from_blender(self, scene):
        for obj in scene.objects:
            _node = SceneNode()
            _node.from_blender(obj)
            self._nodes.append(_node)