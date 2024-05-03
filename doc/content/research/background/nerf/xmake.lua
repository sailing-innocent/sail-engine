-- function add_nerf_content(name)
--     target(name)
--         add_rules("latex-content")
--         add_files(name .. ".tex")
--         add_deps("bib_nerf")

--         add_deps(
--             "fig_nerf_principle",
--             "demo_nerf_dataset",
--         {order = true})
--     target_end()
-- end

-- add_nerf_content("nerf_intro_doc_en")
-- add_nerf_content("nerf_intro_slide_en")
add_content("radiance_field_brief_doc_en")

add_content("nerf_intro_slide_cn", {
    "ray_march_slide_cn",
    "fig_nerf_principle",
    "demo_nerf_dataset",
    "bib_nerf"
})
add_content("nerf_intro_doc_cn", {
    "fig_nerf_principle",
    "bib_nerf"
})