
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
includes("rtow")
-- the cpp 20 not support features (modules ...)
-- includes("cpp20")
-- includes("cpp23")

if get_config("enable_llvm") then 
    includes("k_compiler")
end