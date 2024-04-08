target("use_arxiv")
    add_rules("latex")
    add_files("main.tex")
    add_deps("arxiv")
    -- bib
    add_deps("bib_sample")
    on_load(function (target)
        target:set("latex_main", "main.tex")
    end)
target_end()