import pytest
import time
import mujoco
import mujoco.viewer

@pytest.mark.todo
def test_data_access():
    m = mujoco.MjModel.from_xml_path("assets/model/mjxml/humanoid.xml")
    d = mujoco.MjData(m)
    print()
    print("n_geom: ", m.ngeom)

    print("geom: thigh", m.geom("thigh_left").pos) # array [0, -0.005, -0.17]
    # valid names
    # butt, floor, foot1_left, foot1_right, foot2_left
    # foot2_left, hand_left, hand_right, head, lower_arm_left
    # lower_arm_right, shin_left, shin_right, thigh_left, thigh_right
    # torso, upper_arm_left, upper_arm_right, waist_lower, waist_upper

