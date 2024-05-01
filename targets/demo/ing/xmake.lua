function add_ing_demo(name) 
    add_demo("demo_ing_" .. name, {"SailIng"})
end 

if get_config("enable_vk") then
    add_ing_demo("vk_basic")
    add_ing_demo("vk_hello")
    add_ing_demo("vk_canvas")
    add_ing_demo("vk_scene")
    add_ing_demo("vk_imgui")
end

function add_win_app(name, deps) 
    target(name)
        set_kind("binary")
        add_files(name .. ".cpp")
        set_languages("c++20")
        add_rules("win.sdk.application")
        if (has_config("enable_dx")) then
            add_links("d3d12", "dxgi", "D3DCompiler")
        end
        add_deps(deps)
    target_end()
end

if get_config("enable_dx") then
    add_win_app("demo_ing_dx_pure_win", { "SailIng" })
    add_win_app("demo_ing_dx_pure_app", { "SailIng" })
    add_win_app("demo_ing_dx_triangle_app", { "SailIng" })
end
