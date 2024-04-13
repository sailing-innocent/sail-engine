target("external_imath")
set_kind("shared")
on_load(function (target)
    if is_mode("debug") then 
        target:set("runtimes", "MDd")
        target:set("optimize", "none")
    else
        target:set("runtimes", "MD")
        target:set("optimize", "aggressive")
    end
end)
add_includedirs("src/Imath", "config", {
    public = true 
})
add_files("src/Imath/**.cpp")
add_defines("IMATH_DLL", {public = true})
add_defines("IMATH_EXPORTS")
target_end()