import pytest 

import glob

@pytest.mark.app
def test_glob():
    f = glob.glob(r'C:/*')
    assert len(f) > 0