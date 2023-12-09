option("shipping_one_archive")
    set_default(false)
    set_showmenu(true)
    set_description("Toggle to build modules in one executable file.")
option_end()

rule("sail.module")
    on_load(function (target, opt)
        -- add codegen headers
        -- target_gendir 
        -- jsonfile
        -- embedfile
        -- headerfile: .gen/plat/target/codegen/target_name/module.codegen.h
        -- target dataset
        -- add embed files
    end)
    on_config(function(target)
        -- do generation 
    end)
rule_end()


rule("sail.static_module")
    on_load(function(target, opt) 
        local api = target:extraconf("rules", "sail.static_module", "api")
        target:set("kind", "static")
        if (not has_config("shipping_one_archive")) then 
            target:add("defines", api.."_STATIC", { public = true })
            target:add("defines", api.."_IMPL")
        end
    end)
rule_end()

function shared_module(name, api, version, opt) 
    target(name)
        if has_config("shipping_one_archive") then 
            set_kind("static")
        else 
            set_kind("shared")
        end

        on_load(function(target, opt)
            -- if arxiv
        end)
    target_end()
end 