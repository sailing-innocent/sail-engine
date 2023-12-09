SHARED_MODULE("SailDummy", "SAIL_DUMMY", engine_version)
    add_includedirs("include", {public = true})
    add_files("src/**.cpp")
    add_deps("SailBase")
