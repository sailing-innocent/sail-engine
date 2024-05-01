
if has_config("enable_inno") then
    includes("inno")
end

-- if has_config("enable_ing") then 
--     includes("ing")
-- end 

if has_config("enable_cuda") then 
    includes("cu")
end

if has_config("enable_gl") then 
    includes("gl")
    if has_config("enable_vk") then 
        includes("vk")
    end 
end

if has_config("enable_llvm") then 
    includes("llvm")
end
