add_latex("master_thesis_proposal_slide", {
    "njupre",
    "nju_course_slide_cn",
    "d5_intern_slide_cn",
    "csig2023_slide_cn",
    "research_background_slide_cn",
    "nerf_intro_slide_cn",
    "gaussian_intro_slide_cn"
}, "proposal_slide")

-- target("master_thesis_proposal_slide")
--     add_rules("latex")
--     add_files("**.tex")
--     add_deps("njupre", { order = true})
--     add_deps(
--         "nju_course_slide_cn",
--         "d5_intern_slide_cn",
--         "csig2023_slide_cn",
--         "research_background_slide_cn",
--         
--         "gaussian_intro_slide_cn",
--         -- "content_graphics"
--     {order = true})
--     add_deps(
--         "paper_segment_anything",
--         "fig_impl_20231018",
--         "fig_diff_render_20231018",
--         "fig_panorama_20231018",
--         "fig_reimpl_20231018",
--         "fig_with_gaussian_20231018",
--     {order = true})
--     on_load(function (target)
--         target:set("latex_main", "proposal_slide.tex")
--     end)
-- target_end()