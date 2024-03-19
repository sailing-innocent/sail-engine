function LIBRARY_DEPENDENCY(dep, version, settings)
    add_deps(dep, {public=true})
    add_values(dep .. ".version", version)
end

function PUBLIC_DEPENDENCY(dep, version, settings)
    add_deps(dep, {public=true})
    add_values("sail.module.public_dependencies", dep)
    add_values(dep .. ".version", version)
end

rule("sail.component")
    on_config(function (component, opt)
        import("core.project.project")
        local owner_name = component:extraconf("rules", "sail.component", "owner")
        local owner = project.target(owner_name)
        -- insert owner's include dirs
        for _, owner_inc in pairs(owner:get("includedirs")) do
            component:add("includedirs", owner_inc, {public = true})
        end
        local owner_api = owner:extraconf("rules", "sail.dynamic_module", "api") or owner:extraconf("rules", "sail.static_module", "api")
        -- import api from owner
        component:add("defines", owner_api.."_API=SAIL_IMPORT", owner_api.."_LOCAL=error")
    end)
rule_end()

function STATIC_COMPONENT(name, owner, settings)
    target(owner)
        add_deps(name, { public = opt and opt.public or true })
    target_end()

    target(name)
        set_group("01.modules/"..owner.."/components")
        add_rules("sail.component", { owner = owner })
        set_kind("static")
        
end

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

rule("sail.static_library")
    on_load(function (target, opt)
        target:set("kind", "static")
    end)
rule_end()

rule("sail.static_module")
    add_deps("sail.module")
    add_deps("sail.static_library")
    on_load(function(target, opt)
        local api = target:extraconf("rules", "sail.static_module", "api")
        target:add("defines", api.."_API", {public=true})
        target:add("defines", api.."_STATIC", {public=true})
        target:add("defines", api.."_IMPL")
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


