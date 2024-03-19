target("SailBase")
    set_kind("static")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
    -- eastl
    add_packages("eastl", { public = true})
    -- real time math
    add_deps("external_rtm")
target_end()