add_requires("stb")
target("external_stb_util")
    set_kind("shared")
    add_files("stb_util.cpp")
    add_rules("utils.symbols.export_all")
    add_packages("stb", {public = true})
	on_load(function (target)
		if is_mode("debug") then 
			target:set("runtimes", "MDd")
			target:set("optimize", "none")
		else
			target:set("runtimes", "MD")
			target:set("optimize", "aggressive")
		end
	end)
target_end()