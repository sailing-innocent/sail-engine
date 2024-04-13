target("external_tiny_obj_loader_util")
    set_kind("static")
	add_includedirs("include", { public = true })
	add_files("tiny_obj_loader.cpp")
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