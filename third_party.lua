


rule("link_engine")
    on_load(function(target)
        local engine_path = target:extraconf("rules", "link_engine", "engine_path")
        if engine_path then
            -- add include directories
            -- copy dlls to target directory
            local engine_bin_dir = path.join(engine_path, "bin/release")
            target:add("linkdirs", engine_bin_dir)
            target:add("links", 
                "SailCore", 
            { public = true})
    
        else
            print("engine_path is nil")
        end
    end)

    after_build(function(target))
rule_end()