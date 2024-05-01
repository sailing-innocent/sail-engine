function add_demo_gl(name) 
    add_demo("demo_gl_" .. name, {"SailGL"})
end

add_demo_gl("pure")
add_demo_gl("scene")
add_demo_gl("primitive")
add_demo_gl("curve_fit")