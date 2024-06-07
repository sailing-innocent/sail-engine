SHARED_MODULE("SailVK", "SAIL_VK", engine_version)
    add_includedirs("include", {public=true})
    add_files("src/**.cpp")
    -- based on glfw window
    add_deps("SailGL")
    add_packages("vulkansdk", "imgui", {public = true})
    add_deps("imgui_node_editor", {public = true})