-- function add_mip360_content(name)
--     target(name)
--         add_rules("latex-content")
--         add_files(name .. ".tex")
--         add_deps("figs_demo_mip360")
--         add_deps("demo_colmap_dataset")
--     target_end()
-- end 

add_content("dataset_mip360_intro_doc_cn", {
    "tab_psnr_mip360_sota"
})
add_content("mip360_dataset_intro_slide_cn", {
    "fig_pack_demo_mip360"
})