from .base import SceneObject 
import struct
from module.scene.gltf.data import Buffer as GLTFBuffer 
from module.scene.gltf.data import BufferView as GLTFBufferView
from module.scene.gltf.data import Accessor as GLTFAccessor
from module.scene.gltf.mesh import Mesh as GLTFMesh
from module.scene.gltf.mesh import MeshPrimitive as GLTFMeshPrimitive
from module.scene.gltf.mesh import VertexAttributes as GLTFVertexAttributes

class Mesh(SceneObject):
    def __init__(self):
        self._name = "" # Single Identifier
        self.use_normal = False 
        self.use_tangent = False 
        self.n_tex_coord = 0
    
        self.vert_pos = []
        # TODO fetch normal, tangent and texcoord
        self.vert_normal = []
        self.vert_tangent = []
        self.indices = []
        self._idx = -1
    
    def from_blender(self, obj):
        self._name = obj.name
        self.use_normal = obj.use_normal
        self.use_tangent = obj.use_tangent
        self.n_tex_coord = obj.n_tex_coord
        self.vert_pos = obj.vert_pos
        self.vert_normal = obj.vert_normal
        self.vert_tangent = obj.vert_tangent
        self.indices = obj.indices

    def to_gltf(self):
        accessors = []
        buffer_views = []
        buffer = GLTFBuffer()

        vert_attri = GLTFVertexAttributes()
        vert_attri._has_normal = self.use_normal
        vert_attri._has_tangent = self.use_tangent
        vert_attri._n_tex_coord = self.n_tex_coord
        vert_attri._POSITION = 0
        offset = 1
        vert_bytes = bytes()
        for i, v in enumerate(self.vert_pos):
            vert_bytes += struct.pack('<f', v)
        
        pos_view = GLTFBufferView()
        pos_view._buffer = 0
        pos_view._byte_offset = 0
        pos_view._byte_length = len(vert_bytes)
        pos_view._byte_stride = 16
        pos_view._target = 34962
        buffer_views.append(pos_view)

        pos_accessor = GLTFAccessor()
        pos_accessor._buffer_view = 0
        pos_accessor._byte_offset = 0
        pos_accessor._component_type = 5126
        print(self.vert_pos)
        pos_accessor._count = len(self.vert_pos) // 4
        pos_accessor._type = "VEC3"
        accessors.append(pos_accessor)

        # TODO other views
        if self.use_normal:
            for i, v in enumerate(self.vert_normal):
                vert_bytes += struct.pack('<f', v)
                if (i + 1 ) % 3 == 0:
                    vert_bytes += struct.pack('<f', 1.0)
            offset += 1

        if self.use_tangent:
            for i, v in enumerate(self.vert_tangent):
                vert_bytes += struct.pack('<f', v)
                if (i + 1 ) % 3 == 0:
                    vert_bytes += struct.pack('<f', 1.0)
            offset += 1

        if self.n_tex_coord > 0:
            for i in range(self.n_tex_coord):
                for j, v in enumerate(self.vert_texcoord[i]):
                    vert_bytes += struct.pack('<f', v)
                    if (j + 1 ) % 2 == 0:
                        vert_bytes += struct.pack('<f', 1.0)
            offset += self.n_tex_coord
        
        index_bytes = bytes()
        for i in self.indices:
            index_bytes += struct.pack('<I', i)
    
        index_view = GLTFBufferView()
        index_view._buffer = 0
        index_view._byte_offset = len(vert_bytes)
        index_view._byte_length = len(index_bytes)
        index_view._target = 34963
        buffer_views.append(index_view)

        index_accessor = GLTFAccessor()
        index_accessor._buffer_view = 1
        index_accessor._byte_offset = 0
        index_accessor._component_type = 5125
        index_accessor._count = len(self.indices)
        index_accessor._type = "SCALAR"
        accessors.append(index_accessor)

        primitive = GLTFMeshPrimitive()
        primitive._attributes = vert_attri
        primitive._indices = offset

        # merge bytes
        buffer_bytes = vert_bytes + index_bytes
        # create buffer

        buffer._byte_length = len(buffer_bytes)
        buffer._uri = self._name + ".bin"
        buffer._data = buffer_bytes 


        mesh = GLTFMesh()
        mesh._primitives.append(primitive)
        mesh._extras = {
            "bin": buffer._uri,
            "file_size": 272,
            "vertex_count": 8,
            "normal": False,
            "tangent": False,
            "uv_count": 0,
            "submesh_offset": [
                0
            ]
        }

        return mesh, accessors, buffer_views, buffer