if get_config("enable_inno") then
    includes("innopy")
end
if get_config("enable_cuda") then 
    includes("sailcupy")
end 
