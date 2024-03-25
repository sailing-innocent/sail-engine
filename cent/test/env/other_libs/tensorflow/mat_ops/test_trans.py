from utils.transform.tf_utils import trans_t

def test_trans_t():
    mat = trans_t(1)
    assert mat.shape == (4, 4)
