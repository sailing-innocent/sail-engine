target("SailBase")
    set_kind("static")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
    add_packages("eastl", { public = true})
target_end()