
rule("sail.module")
    on_load(function (target, opt)
        if is_mode("debug") then
            target:set("runtimes", "MDd")
            target:set("optimize", "none")
            target:set("warnings", "none")
            target:add("cxflags", "/GS", "/Gd", {
                tools = {"clang_cl", "cl"}
            })
            target:add("cxflags", "/Zc:preprocessor", {
                tools = "cl"
            });
        else
            target:set("runtimes", "MD")
            target:set("optimize", "aggressive")
            target:set("warnings", "none")
            target:add("cxflags", "/GS-", "/Gd", {
                tools = {"clang_cl", "cl"}
            })
            target:add("cxflags", "/Zc:preprocessor", {
                tools = "cl"
            })
        end 
        -- use simd
        target:add("vectorexts", "avx", "avx2")
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
    after_build(function(target)
    end)
rule_end()



rule("sail.dynamic_module")
    add_deps("sail.module")
    on_load(function(target, opt)
        local api = target:extraconf("rules", "sail.dynamic_module", "api")
        local version = target:extraconf("rules", "sail.dynamic_module", "version")
        target:add("defines", api.."_API=SAIL_IMPORT", {public=true})
        target:add("defines", api.."_API=SAIL_EXPORT", {public=false})
    end)
rule_end()

function SHARED_MODULE(name, api, version, opt) 
    target(name)
        set_kind("shared")
        set_languages("clatest", "c++20")
        set_exceptions("cxx")
        set_group("01.modules/"..name)
        add_rules("sail.dynamic_module", {api=api, version=version})

end


