SHARED_MODULE("SailGL", "SAIL_GL", engine_version)
    add_includedirs("include", {public=true})
    add_files("src/**.cpp")
    add_deps("SailDummy")
    add_packages("glfw", "imgui", {public = true})
    add_deps("external_glad", {public = true})