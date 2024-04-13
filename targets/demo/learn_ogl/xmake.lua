target("learn_ogl_util")
    set_kind("static")
    add_files("util.cpp")
    set_languages("c++20")
    add_packages("glm", "glfw", {public = true})
    add_deps("external_glad", "external_stb_util", {public = true})
target_end()

function add_ogl_demo(name) 
    add_demo("demo_ogl_" .. name, {"learn_ogl_util"})
end

add_ogl_demo("00.plain_window")
add_ogl_demo("01.triangle")
add_ogl_demo("02.rectangle")
add_ogl_demo("03.texture")
add_ogl_demo("04.transform")