target("use_arxiv")
    add_rules("latex", {latex_compiler = "pdflatex"})
    add_files("main.tex")
    add_deps("arxiv")
    -- bib
    add_deps("bib_sample")
target_end()
