SHARED_MODULE("SailCore", "SAIL_CORE", engine_version)
    add_includedirs("include", {public = true})
    add_files("src/**.cpp")
    add_deps("SailBase")