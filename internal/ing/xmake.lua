add_requires("glfw")
add_requires("glm")

SHARED_MODULE("SailIng", "SAIL_ING", engine_version)
    add_includedirs("include", {public=true})
    add_files("src/**.cpp")
    add_deps("SailBase")
    add_packages("glm", "glfw", {public = true})
    add_deps("external_glad", {public = true})