from .mission import Mission 


def test_profile():
    # f = "profile_vanilla_train"
    # f = "profile_vanilla_eval"
    # f = "profile_vanilla_diff_init_train"
    f = "profile_vanilla_diff_trainer"
    m = Mission(f + ".json")
    m.exec()
