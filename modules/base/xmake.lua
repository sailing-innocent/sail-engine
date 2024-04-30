
target("SailBase")
    set_kind("static")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
    add_packages("glm", { public = true})
target_end()