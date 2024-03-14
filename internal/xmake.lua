
if has_config("enable_inno") then
    includes("inno")
end

if has_config("enable_ing") then 
    includes("ing")
end 

if has_config("enable_cuda") then 
    includes("cu")
end