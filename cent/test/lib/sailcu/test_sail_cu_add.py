import pytest 

import sys 
import os 
cwd = os.path.join(os.getcwd(), "../bin/release")
sys.path.append(cwd)

from sailcupy import add

@pytest.mark.current 
def test_ing_add():
    assert add(1, 2) == 3