set_xmakever("2.8.5")

set_project("SailEngine")

engine_version = "0.1.0"

-- default options
includes("build_files/xmake/default_options.lua")
-- try generated options
includes("build_files/xmake/_options.lua")
-- rules 
includes("build_files/xmake/rules.lua")

-- modules
includes("external") -- external dependencies
includes("source") -- core engine
includes("internal") -- internal modules
includes("tests") -- tests