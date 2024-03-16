import pytest 

from module.blender.wrapper import blender_executive
from module.blender.model.armature.double_bone import DoubleBone
import bpy
import numpy as np 

@blender_executive
def double_bone_ik(rootdir):
    total_frames = 30
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = total_frames
    double_bone = DoubleBone()
    N_keyframes = 5
    target_height = 1
    max_offset = np.sqrt(3)
    pi = 3.1415926
    # initial
    # here we only consider the rotation of z axis (x,y) plane
    theta_1 = pi / 3
    # inverse calcuate next bone's rotation
    theta_2 = 0
    bone_1 = double_bone.bone_1()
    bone_2 = double_bone.bone_2()
    
    bpy.ops.object.mode_set(mode='POSE')

    for i in range(N_keyframes):
        frame = i * total_frames // N_keyframes
        target_end_position = (0, i / N_keyframes * 2 * max_offset - max_offset, target_height)

        steps = 100
        lr = 0.1
        for j in range(steps):
            end_position = (
                0,
                - bone_1.length * np.sin(theta_1) - bone_2.length * np.sin(theta_1 + theta_2),
                bone_1.length * np.cos(theta_1) + bone_2.length * np.cos(theta_1 + theta_2)
            )
            # inverse kinematics
            # Jacobian
            J = np.zeros((2, 2))
            J[0, 0] = - bone_1.length * np.cos(theta_1) - bone_2.length * np.cos(theta_1 + theta_2)
            J[1, 0] = - bone_2.length * np.cos(theta_1 + theta_2)
            J[0, 1] = - bone_1.length * np.sin(theta_1) - bone_2.length * np.sin(theta_1 + theta_2)
            J[1, 1] = - bone_2.length * np.sin(theta_1 + theta_2)
            # delta
            delta = np.zeros(2)
            delta[0] = target_end_position[1] - end_position[1]
            delta[1] = target_end_position[2] - end_position[2]
            # print("delta: ", delta)
            # update theta
            delta_theta = lr * np.linalg.pinv(J) @ delta
            # print("delta_theta: ", delta_theta)
            theta_1 = theta_1 + delta_theta[0]
            theta_2 = theta_2 + delta_theta[1]
            # clamp
            theta_1 = min(theta_1, 2 * pi)
            theta_1 = max(theta_1, 0)
            theta_2 = min(theta_2, 2 * pi )
            theta_2 = max(theta_2, 0)
            # update bone
        bone_1.rotation_euler = (0, 0, theta_1)
        bone_2.rotation_euler = (0, 0, theta_2)

        bone_1.keyframe_insert(data_path='rotation_euler', frame=frame)
        # inverse calcuate next bone's rotation
        bone_2.keyframe_insert(data_path='rotation_euler', frame=frame)
    
    # return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

@pytest.mark.current 
def test_double_bone_ik():
    double_bone_ik("animation", "double_bone_ik", clear=True)
    assert True 