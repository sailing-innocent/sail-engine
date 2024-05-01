
if get_config("enable_gl") then 
    add_requires("glfw")
end 

if get_config("enable_vk") then 
    add_requires("glfw")
    add_requires("vulkansdk", {optional = true})
end 

SHARED_MODULE("SailIng", "SAIL_ING", engine_version)
    add_includedirs("include", {public=true})
    add_files("src/**.cpp")
    
    add_deps("SailBase")

    if get_config("enable_cuda") then
        add_deps("SailCu")
        add_defines("SAIL_ING_CUDA", {public=true})
    end

    if get_config("enable_gl") then 
        add_packages("glm", "glfw", "imgui", {public = true})
        add_deps("external_glad", {public = true})
        add_defines("SAIL_ING_GL", {public=true})
    end 

    if get_config("enable_vk") then 
        add_packages("glfw", "imgui", "vulkansdk", {public = true})
    end
    if (has_config("enable_dx")) then
        add_links("d3d12", "dxgi", "D3DCompiler", {public = true})
    end

    add_deps("external_stb_util", {public = true})
    add_deps("external_tiny_obj_loader_util", {public = true})