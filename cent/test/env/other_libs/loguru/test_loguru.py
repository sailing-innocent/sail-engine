import pytest 

from loguru import logger 

@pytest.mark.current 
def test_loguru():
    logger.debug("That's it")