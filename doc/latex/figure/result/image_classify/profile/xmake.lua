
add_dat("result_simple_mlp_batchsize_32_data")
add_dat("result_simple_mlp_batchsize_64_data")
target("pgf_profile_img_classify_mlp_bs_ep_result")
    add_rules("latex-content")
    add_files("pgf_profile_img_classify_mlp_bs_ep_result.tex")
    add_deps(
"result_simple_mlp_batchsize_32_data",
"result_simple_mlp_batchsize_64_data"
)
target_end()
        