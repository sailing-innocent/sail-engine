target("external_imath")
set_kind("shared")
add_includedirs("src/Imath", "config", {
    public = true 
})
add_files("src/Imath/**.cpp")
add_defines("IMATH_DLL", {public = true})
add_defines("IMATH_EXPORTS")
target_end()