from .mission import Mission 


def test_profile():
    f = "profile_vanilla_train"
    m = Mission(f + ".json")
    m.exec()
    # assert True 