SHARED_MODULE("SailScene", "SAIL_SCENE", engine_version)
    add_includedirs("include", {public = true})
    add_files("src/**.cpp")
    add_deps("SailCore")