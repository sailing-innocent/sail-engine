import pytest 
import sys 
import os 
sys.path.append(os.path.join(os.getcwd(), "../bin/release"))
from innopy import add

@pytest.mark.current 
def test_add():
    assert add(1,2) == 3