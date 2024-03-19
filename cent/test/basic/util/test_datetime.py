import pytest 

import datetime

@pytest.mark.current 
def test_datetime():
    today = datetime.date.today()
    day100 = datetime.timedelta(days=200)
    print(today + day100)