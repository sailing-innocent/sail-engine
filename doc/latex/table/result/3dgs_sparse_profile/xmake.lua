add_json_table("tab_3dgs_profile_diff_init_psnr_result_nerf_blender")
add_json_table("tab_3dgs_profile_basic_diff_init_psnr_result_nerf_blender")

target("tab_pack_sparse_3dgs_profile_result")
    set_kind("phony")
    add_deps(
        "tab_3dgs_profile_diff_init_psnr_result_nerf_blender",
        "tab_3dgs_profile_basic_diff_init_psnr_result_nerf_blender"
    )

target_end()