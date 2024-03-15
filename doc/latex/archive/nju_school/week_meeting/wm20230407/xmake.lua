target("bib_20230407")
    add_rules("latex-bib")
    add_files("bib_20230407.bib")
    add_deps("bib_diffusion_model")
target_end()

target("wm20230407")
    add_rules("latex")
    add_files("pre.tex")
    -- for fig pack
    add_deps("fig_wm20230407", { order = true })
    -- for bib
    add_deps("bib_20230407")
    -- for content 
    add_deps(
        "gan_intro_slide_en",
        "sam_intro_slide_en",
        "fluid_simulation_intro_slide_en"
    )
    on_load(function (target)
        target:set("latex_main", "pre.tex")
    end)
target_end()