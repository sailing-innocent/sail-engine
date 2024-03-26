from .mission import Mission 


def test_profile():
    # f = "profile_vanilla_train"
    f = "profile_vanilla_eval"
    m = Mission(f + ".json")
    m.exec()
    # assert True 