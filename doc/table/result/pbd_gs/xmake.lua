add_json_table("tab_psnr_mip360_pbd_gs")
add_json_table("tab_psnr_ns_pbd_gs")
add_json_table("tab_ssim_mip360_pbd_gs")

target("tabs_pbd_gs_compare_result")
    set_kind("phony")
    add_deps({
        "tab_psnr_mip360_pbd_gs",
        "tab_psnr_ns_pbd_gs",
        "tab_ssim_mip360_pbd_gs"
    })
target_end()