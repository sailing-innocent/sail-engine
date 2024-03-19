import bpy
import random
from mathutils import Vector

def create_random_metaball(origin=(0,0,0), n=30, r0=4, r1=2.5):
    metaball = bpy.data.metaballs.new('Metaball')
    obj = bpy.data.objects.new('MetaballObject', metaball)
    bpy.context.collection.objects.link(obj)

    metaball.resolution = 0.2
    metaball.render_resolution = 0.05

    for i in range(n):
        location = Vector(origin) + Vector(random.uniform(-r0, r0) for i in range(3))
        element = metaball.elements.new()
        element.co = location
        element.radius = r1

    return obj
