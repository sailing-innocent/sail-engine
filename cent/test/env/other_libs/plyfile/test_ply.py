import pytest 

from plyfile import PlyData, PlyElement
import numpy as np 

@pytest.mark.app
def test_read_ply():
    plydata = PlyData.read('./asset/model/ply/tet.ply')
    assert plydata.elements[0].name == 'vertex'
    print(plydata.elements[0].data) # [(0,0,1)...]
    print(plydata.elements[1].data) # [array(0,1,2),255,255,255]

@pytest.mark.current 
def test_write_ply():
    ply_path = './asset/model/ply/wrt.ply'
    vertex = np.array([(0,0,0),(0,1,1),(1,0,1),(1,1,0)],dtype=[('x','f4'),('y','f4'),('z','f4')])

    face = np.array([([0,1,2],255,255,255),([0,2,3],255,0,0),([0,1,3], 0, 255, 0), ([1,2,3], 0, 0, 255)], dtype=[('vertex_indices','i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el_vert = PlyElement.describe(vertex, 'vertex', comments=['this is vertex comment'])
    el_face = PlyElement.describe(face, 'face',
                    val_types={'vertex_indices': 'u2'},
                    len_types={'vertex_indices': 'u4'})

    PlyData([el_vert, el_face], text=True).write(ply_path)

    assert True 