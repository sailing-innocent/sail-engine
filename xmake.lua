set_xmakever("2.9.2")
add_rules("mode.release", "mode.debug")
engine_version = "0.1.0"
add_repositories("sail-repo xrepo")

set_toolchains("clang-cl")
set_exceptions("cxx")
set_runtimes("MD")
add_cxflags("/GS", "/Gd", { 
    tools = {"clang-cl", "cl"}
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
-- external dependencies
if get_config("enable_vk") then
    add_requires("vulkansdk")
end
-- self xrepo dependencies
add_requires("glm", {version = "2024.05.15-sail"}) 
add_requires("eastl", {version = "2024.05.15-sail"})
add_requires("imgui", {version = "2024.05.27-sail"}) 
add_requires("glfw", {configs = {vulkan = true}})

-- self dependencies
set_languages("c++20")
includes("external") -- external dependencies
includes("modules") -- core engine
includes("internal") -- internal independent extensions
-- targets
includes("tests") -- tests
includes("targets")
-- documentation
if get_config("enable_doc") then
    includes("doc") -- documentation
end