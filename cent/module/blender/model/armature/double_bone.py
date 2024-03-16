import bpy 
from typing import Tuple

class DoubleBone:
    def __init__(self, root: Tuple[float] = (0, 0, 0), bone_1_name: str = "bone_1", bone_1_length: float = 1.0, bone_2_name: str = "bone_2", bone_2_length: float = 1.0):
        self.bone_1_name = bone_1_name
        self.bone_1_length = bone_1_length
        self.bone_2_name = bone_2_name
        self.bone_2_length = bone_2_length  

        bpy.ops.object.armature_add(enter_editmode=True)
        self.armature = bpy.context.object
        bpy.ops.object.mode_set(mode='EDIT')
        bone = self.armature.data.edit_bones['Bone']
        bone.name = bone_1_name
        bone.head = root
        bone.tail = (root[0], root[1], root[2] + bone_1_length)
        # add bone
        next_bone = self.armature.data.edit_bones.new(bone_2_name)
        next_bone.head = bone.tail
        next_bone.tail = (root[0], root[1], root[2] + bone_1_length + bone_2_length)
        # hierarchy
        next_bone.parent = bone

        bpy.ops.object.mode_set(mode='OBJECT')
        self.bone_1().rotation_mode = 'XYZ'
        self.bone_2().rotation_mode = 'XYZ'

    def bone_1(self):
        return self.armature.pose.bones[self.bone_1_name]
    
    def set_bone_1_rotation(self, rotation: Tuple[float]):
        bpy.ops.object.mode_set(mode='POSE')
        self.bone_1().rotation_euler = rotation
        bpy.ops.object.mode_set(mode='OBJECT')

    def bone_2(self):
        return self.armature.pose.bones[self.bone_2_name]
    
    def set_bone_2_rotation(self, rotation: Tuple[float]):
        bpy.ops.object.mode_set(mode='POSE')
        self.bone_2().rotation_euler = rotation
        bpy.ops.object.mode_set(mode='OBJECT')

    def obj(self):
        return self.armature