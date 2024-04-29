SHARED_MODULE("SailRT", "SAIL_RT", engine_version)
    add_includedirs("include", {public = true})
    add_files("src/**.cpp")
    add_deps("SailCore")