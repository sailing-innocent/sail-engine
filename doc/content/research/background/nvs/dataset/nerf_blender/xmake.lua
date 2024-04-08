function add_nerf_blender_content(name)
    target(name)
        add_rules("latex-content")
        add_files(name .. ".tex")
        add_deps("figs_demo_nerf_blender", {order=true})
    target_end()
end 

add_nerf_blender_content("nerf_blender_dataset_intro_doc_en")
add_nerf_blender_content("nerf_blender_dataset_intro_slide_cn")