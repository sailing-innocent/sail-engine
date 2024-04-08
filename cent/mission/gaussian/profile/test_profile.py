from .mission import Mission 


def test_profile():
    # f = "profile_vanilla_train"
    # f = "profile_vanilla_eval"
    # f = "profile_vanilla_diff_init_train"
    # f = "profile_vanilla_diff_tp_train"
    # f = "profile_mip360_train"
    # f = "profile_mip360_eval"
    f = "profile_vanilla_diff_train_loss_mip360_train"
    m = Mission(f + ".json")
    m.exec()
