target("njuthesis")
    add_rules("latex.template")
    add_files("*.dtx", "*.bst", "*.cfg", "*.cls", "*.sty","*.png", "*.jpg")
target_end()