

function add_gaussian_splatting_content(name)
    add_content(name, {
        "fig_tile_based_raster",
        "fig_surface_splatting",
        "fig_splatting",
        "fig_tile_based_renderer_impl",
        "fig_demo_gaussian_mixture_gaussian",
        "fig_demo_guassian_3d_gaussian",
        "fig_demo_res_img_gaussian",
        "fig_demo_em_algorithm",
        "bib_gaussian"
    })
end 

add_gaussian_splatting_content("gaussian_intro_doc_cn")
add_gaussian_splatting_content("gaussian_intro_doc_en")
add_gaussian_splatting_content("gaussian_intro_slide_cn")
add_gaussian_splatting_content("gaussian_em_slide_cn")

target("gaussian_update_slide_cn")
    add_rules("latex-content")
    add_files("gaussian_update_slide_cn.tex")
    add_deps("bib_gaussian")
target_end()