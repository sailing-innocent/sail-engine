set_xmakever("2.8.7")
set_project("SailEngine")
engine_version = "0.1.0"
add_rules("mode.release", "mode.debug")
set_toolchains("clang-cl")
if is_mode("debug") then
    set_targetdir("bin/debug")
else
    set_targetdir("bin/release")
end

-- default options
includes("xmake/default_options.lua")
-- rules 
includes("xmake/rules.lua")
-- modules
includes("external") -- external dependencies
includes("modules") -- core engine
includes("tests") -- tests
