set_xmakever("2.9.1")
set_project("SailEngine")
engine_version = "0.1.0"
add_rules("mode.release", "mode.debug")
set_toolchains("clang-cl")
set_exceptions("cxx")
set_runtimes("MD")
add_cxflags("/GS", "/Gd", {
    tools = {"clang_cl", "cl"}
})
add_cxflags("cxflags", "/Zc:preprocessor", {
    tools = "cl"
});

if is_mode("debug") then
    set_targetdir("bin/debug")
else
    set_targetdir("bin/release")
end

-- default options
includes("xmake/default_options.lua")
-- rules 
includes("xmake/rules.lua")

-- assets 
includes("assets") 

-- external and requirements
add_requires("glm") -- for math calculation
add_requires("glfw", {
    configs = {
        vulkan = true
    }
}) -- for window management
add_requires("imgui", {configs = {glfw_opengl3 = true, vulkan = true}}) -- for ui
if get_config("enable_vk") then
    add_requires("vulkansdk", {optional = true})
end
includes("external") -- external dependencies

-- internal and core modules
includes("modules") -- core engine
includes("internal") -- internal independent extensions

-- targets
includes("tests") -- tests
includes("targets")

-- documentation
if get_config("enable_doc") then
    includes("doc") -- documentation
end