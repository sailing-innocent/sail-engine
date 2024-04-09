import pytest 

from app.diff_tools.gs_sh_processor.torch_ext import GaussianSHProcessor

@pytest.mark.current 
def test_torch_ext_sh_processor():
    processor = GaussianSHProcessor()
    assert True