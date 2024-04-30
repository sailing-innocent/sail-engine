local function sail_add_test(folder, name, deps)
    target("test_" .. folder .. "_" .. name)
        set_kind("binary")
        on_load(function (target)
            target:set("languages", "c++20")
            target:set("exceptions", "cxx")
            if is_mode("debug") then 
                target:set("runtimes", "MDd")
                target:set("optimize", "none")
            else
                target:set("runtimes", "MD")
                target:set("optimize", "aggressive")
            end
        end)
        add_deps("external_doctest")
        local match_str = path.join(name, "**.cpp")
        add_includedirs("_framework")
        add_files(path.join("_framework/test_util.cpp"), path.join(folder, match_str))
        for _, dep in ipairs(deps) do
            add_deps(dep)
        end
    target_end()
end
sail_add_test("basic", "semantic", {})
sail_add_test("basic", "stl", {})
sail_add_test("basic", "io", {
    "external_tiny_obj_loader_util"
})
sail_add_test("basic", "leetcode", {})
-- sail_add_test("basic", "dummy", {"SailDummy"})

-- Core
-- sail_add_test("core", "math", {"SailCore"})

if has_config("enable_inno") then 
    sail_add_test("inno", "util", {"SailInno"})
    sail_add_test("inno", "helper", {"SailInno"})
end 
if has_config("enable_ing") then 
    sail_add_test("ing", "util", {"SailIng"})
    if has_config("enable_dx") then 
        sail_add_test("ing", "dx", {"SailIng"})
    end
end 
if has_config("enable_cuda") then 
    sail_add_test("cu", "util", {"SailCu"})
end 

if has_config("enable_gl") then 
    -- sail_add_test("gl", "util", {"SailGL"})
end

if has_config("enable_llvm") then 
    sail_add_test("llvm", "ast", {"SailLLVM"})
end 