add_requires("glfw")
add_requires("vulkansdk")
target("imgui")
    set_kind("static")
    add_packages("glfw")
    add_packages("vulkansdk")
    add_headerfiles("imgui/(**.h)")
    add_includedirs("imgui", {public=true})
    add_includedirs("imgui/backends", {public=true})
    add_files(
        "imgui/imgui.cpp", 
        "imgui/imgui_tables.cpp", 
        "imgui/imgui_widgets.cpp", 
        "imgui/imgui_draw.cpp", 
        "imgui/imgui_demo.cpp", 
    -- glfw
    "imgui/backends/imgui_impl_glfw.cpp", 
    "imgui/backends/imgui_impl_opengl3.cpp",
    -- vulkan
    "imgui/backends/imgui_impl_vulkan.cpp",
    -- win32
    "imgui/backends/imgui_impl_win32.cpp",
    -- dx12
    "imgui/backends/imgui_impl_dx12.cpp")

