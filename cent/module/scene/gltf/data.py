from .base import GLTFObject

class Buffer(GLTFObject):
    def __init__(self):
        self._uri = None 
        self._byte_length = None
        self._data = None 

    def from_gltf(self, gltf_json):
        self._uri = gltf_json["uri"]
        self._byte_length = gltf_json["byteLength"]
        # TODO read bytes from uri

    def to_gltf(self):
        obj = {}
        obj["uri"] = self._uri
        obj["byteLength"] = self._byte_length
        return obj, self._data

class BufferView(GLTFObject):
    def __init__(self):
        self._buffer = None
        self._byte_offset = None
        self._byte_length = None
        self._byte_stride = None
        self._target = None 

    def from_gltf(self, gltf_json):
        self._buffer = gltf_json["buffer"]
        self._byte_offset = gltf_json["byteOffset"]
        self._byte_length = gltf_json["byteLength"]
        if "byteStride" in gltf_json:
            self._byte_stride = gltf_json["byteStride"]
        self._target = gltf_json["target"]

    def to_gltf(self):
        obj = {}
        obj["buffer"] = self._buffer
        obj["byteOffset"] = self._byte_offset
        obj["byteLength"] = self._byte_length
        if self._byte_stride:
            obj["byteStride"] = self._byte_stride
        obj["target"] = self._target
        return obj

class Accessor(GLTFObject):
    def __init__(self):
        self._buffer_view = None
        self._byte_offset = None
        self._component_type = None
        self._count = None
        self._type = None 
        self._max = None
        self._min = None

    def from_gltf(self, gltf_json):
        self._buffer_view = gltf_json["bufferView"]
        self._byte_offset = gltf_json["byteOffset"]
        self._component_type = gltf_json["componentType"]
        self._count = gltf_json["count"]
        self._type = gltf_json["type"]
        if "max" in gltf_json:
            self._max = gltf_json["max"]    
        if "min" in gltf_json:
            self._min = gltf_json["min"]

    def to_gltf(self):
        obj = {}
        obj["bufferView"] = self._buffer_view
        obj["byteOffset"] = self._byte_offset
        obj["componentType"] = self._component_type
        obj["count"] = self._count
        obj["type"] = self._type
        if self._max:
            obj["max"] = self._max
        if self._min:
            obj["min"] = self._min
        return obj

class Image(GLTFObject):
    def __init__(self, uri):
        self._uri = uri
        self._width = 256
        self._height = 256
        self._data = None 

    def from_gltf(self, gltf_json):
        self._uri = gltf_json["uri"]
        # TODO read img data from uri

    def to_gltf(self):
        obj = {}
        obj["uri"] = self._uri
        # TODO save img data to uri
        return obj

