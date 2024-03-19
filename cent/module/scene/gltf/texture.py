from .base import GLTFObject

class Sampler(GLTFObject):
    def __init__(self, mag_filter, min_filter, wrap_s, wrap_t):
        self._mag_filter = mag_filter
        self._min_filter = min_filter
        self._wrap_s = wrap_s
        self._wrap_t = wrap_t

    def from_gltf(self, gltf_json):
        self._mag_filter = gltf_json["magFilter"]
        self._min_filter = gltf_json["minFilter"]
        self._wrap_s = gltf_json["wrapS"]
        self._wrap_t = gltf_json["wrapT"]


    def to_gltf(self):
        obj = {}
        obj["magFilter"] = self._mag_filter
        obj["minFilter"] = self._min_filter
        obj["wrapS"] = self._wrap_s
        obj["wrapT"] = self._wrap_t
        return obj

class Texture(GLTFObject):
    def __init__(self, sampler_idx, source_idx):
        self._sampler_idx = sampler_idx
        self._source_idx = source_idx

    def to_gltf(self):
        obj = {}
        obj["sampler"] = self._sampler_idx
        obj["source"] = self._source_idx
        return obj