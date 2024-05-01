-- includes lc
local lc_path = get_config("lc_path")
if lc_path then
    includes(path.join(lc_path, "config", "xmake_rules.lua"))
end
includes("thirdparty")
includes("sdks")