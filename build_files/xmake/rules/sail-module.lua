option("shipping_one_archive")
    set_default(false)
    set_showmenu(true)
    set_description("Toggle to build modules in one executable file.")
option_end()

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