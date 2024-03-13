target("shaders")
    set_kind("object")
    add_files("**.vert", "**.frag", "**.hlsl")
    add_rules("sail.asset")
target_end()

includes("vulkan")