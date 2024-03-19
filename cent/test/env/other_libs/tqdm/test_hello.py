from tqdm import tqdm 

def test_hello_tqdm():
    j = 0
    for i in tqdm(range(100)):
        j = i # just do something as the normal for loop

    assert True 