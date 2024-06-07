SHARED_MODULE("SailInno", "SAIL_INNO", engine_version)
    add_includedirs("include", {public = true})
    add_files("src/**.cpp")
    load_lc()
    if get_config("enable_cuda") then 
        add_rules("sail.cuda", "lc.lcub")
    end 
    
    add_deps("SailBase")
