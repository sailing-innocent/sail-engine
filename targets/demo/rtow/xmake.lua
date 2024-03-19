includes("demo_rtow")

function add_rtow_demo(name)
    add_demo("demo_rtow_" .. name, {"demo_rtow"})
end

add_rtow_demo("01.write_image")
add_rtow_demo("02.simple_ray")
add_rtow_demo("03.hit_sphere")
add_rtow_demo("04.antialiasing")
add_rtow_demo("05.sphere_normal")
add_rtow_demo("06.hittable_world")
add_rtow_demo("07.matte")
add_rtow_demo("08.lambertian")
add_rtow_demo("09.materials")
add_rtow_demo("10.dielectrics")
add_rtow_demo("11.advanced_camera")
add_rtow_demo("12.final_draw")
