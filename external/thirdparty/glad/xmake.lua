target("external_glad")
    set_kind("static")
    add_files("src/glad.c")
    add_includedirs("include", {public = true})
target_end()