add_requires("stb")
target("external_stb_util")
    set_kind("static")
    add_files("stb_util.cpp")
    add_packages("stb", {public = true})
target_end()