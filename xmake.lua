set_xmakever("2.8.5")

set_project("sail-engine")

-- default options
includes("build_files/xmake/default_options.lua")
-- try generated options
includes("build_files/xmake/_options.lua")

includes("external") -- external dependencies
includes("source") -- core engine
includes("internal") -- internal modules
includes("tests") -- tests