if get_config("enable_gl") then 
    add_requires("glfw")
    add_requires("glm")
end 
SHARED_MODULE("SailIng", "SAIL_ING", engine_version)
    add_includedirs("include", {public=true})
    add_files("src/**.cpp")
    add_deps("SailBase")
    if get_config("enable_gl") then 
        add_packages("glm", "glfw", {public = true})
        add_deps("external_glad", {public = true})
    end 
    add_deps("external_stb_util", {public = true})