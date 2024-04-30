
target("SailBase")
    set_kind("static")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
    -- eastl
    add_packages("eastl", "glm", { public = true})
    -- real time math
target_end()