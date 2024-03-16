target("shaders")
    set_kind("object")
    add_files("**.vert", "**.frag", "**.hlsl")
    add_rules("sail.asset")
target_end()

if get_config("enable_vk") then 
    includes("vulkan")
end