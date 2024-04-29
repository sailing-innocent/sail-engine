STATIC_COMPONENT("SailGuid", "SailCore")
    set_optimize("fastest")
    LIBRARY_DEPENDENCY("SailBase", engine_version)
    add_files("src/guid/build.*.cpp")

SHARED_MODULE("SailCore", "SAIL_CORE", engine_version)
    add_includedirs("include", {public = true})
    add_files("src/core/build.*.cpp")
    add_deps("SailBase")