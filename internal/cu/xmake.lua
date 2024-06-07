SHARED_MODULE("SailCu", "SAIL_CU", engine_version)
    add_includedirs("include", { public = true })
    add_files("src/**.cu")
    set_languages("c++17")
    add_deps("SailBase")
    add_packages("glm", { public = true })
    add_rules("sail.cuda")