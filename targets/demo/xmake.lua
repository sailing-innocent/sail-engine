if get_config("enable_gl") then 
    includes("learn_ogl")
end

if get_config("enable_inno") then 
    includes("inno")
end 

if get_config("enable_ing") then 
    includes("ing")
end 

includes("ray_march")
