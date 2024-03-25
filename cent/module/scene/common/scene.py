from .base import SceneObject 
from module.scene.gltf.scene import Scene as GLTFScene
from module.scene.gltf.scene import SceneGraph as GLTFSceneGraph
from module.scene.gltf.scene import SceneNode as GLTFSceneNode
from module.scene.blender.scene import Scene as BlenderScene
from .mesh import Mesh 

class Scene(SceneObject):
    def __init__(self):
        self._nodes = []
        self._meshes = {} # name-mesh map 
        self._materials = {} # name-material map
        self._buffer_views = []
        self._accessors = []
        self._buffers = []

    def to_gltf(self):
        scene = GLTFScene()
        # generate mesh map
        for node in self._nodes:
            if node._mesh:
                try:
                    self._meshes[node._mesh._name]
                except KeyError:
                    self._meshes[node._mesh._name] = node._mesh
        
        buffer_offset = 0
        buffer_view_offset = 0
        accessor_offset = 0

        mesh_list = []
        idx = 0
        for mesh in self._meshes.values():
            gltf_mesh, accessors, buffer_views, buffer = mesh.to_gltf()
            gltf_mesh._name = mesh._name
            for prim in gltf_mesh._primitives:
                prim._attributes.offset_all(accessor_offset)
            mesh_list.append(gltf_mesh)
            mesh._idx = idx
            idx += 1
            for accessor in accessors:
                accessor._buffer_view = buffer_view_offset + accessor._buffer_view
                self._accessors.append(accessor)
                accessor_offset += 1
    
            for buffer_view in buffer_views:
                buffer_view._buffer = buffer_offset + buffer_view._buffer
                self._buffer_views.append(buffer_view)
                buffer_view_offset += 1

            self._buffers.append(buffer)
            buffer_offset += 1

        scene._meshes = mesh_list
        scene._buffer_views = self._buffer_views
        scene._accessors = self._accessors
        scene._buffers = self._buffers

        for node in self._nodes:
            if node._mesh:
                gltf_node = node.to_gltf()
                gltf_node._mesh = self._meshes[node._mesh._name]._idx
                scene._nodes.append(gltf_node)
        n_node = len(scene._nodes)
        scene._scene_graph._nodes = [i for i in range(n_node)]

        return scene

    def from_blender(self, scene: BlenderScene):
        n_nodes = len(scene._nodes)
        for i in range(n_nodes):
            node = SceneNode()
            node.from_blender(scene._nodes[i])
            self._nodes.append(node)

class SceneGraph(SceneObject):
    def __init__(self):
        self._nodes = []

class SceneNode(SceneObject):
    def __init__(self):
        self._children = []
        self._mesh = None 
        self._skin = None 
        self._camera = None 
        self._transform_type = "MATRIX"
        self._matrix = None 
        self._translation = None 
        self._rotation = None 
        self._scale = None

    def from_blender(self, obj):
        self._matrix = obj._matrix 
        if obj._type == 'MESH':
            self._mesh = Mesh()
            self._mesh.from_blender(obj._mesh)
        elif obj._type == 'CAMERA':
            return
        elif obj._type == 'LIGHT':
            return
        else:
            print("get unknown")
            print(obj.name)

    def to_gltf(self):
        gltf_node = GLTFSceneNode()
        gltf_node._mesh = -1
        gltf_node._matrix = self._matrix
        return gltf_node