local function sail_add_test(folder, name, deps)
    target("test_" .. folder .. "_" .. name)
        set_kind("binary")
        set_languages("c++20")
        set_exceptions("cxx")
        add_deps("external_doctest")
        local match_str = path.join(name, "**.cpp")
        add_includedirs("framework")
        add_files(path.join("framework/test_util.cpp"), path.join(folder, match_str))
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

sail_add_test("basic", "dummy", {"SailDummy"})
sail_add_test("basic", "deps", {"SailDummy"})

if get_config("enable_inno") then 
    sail_add_test("inno", "util", {"SailInno"})
    sail_add_test("inno", "helper", {"SailInno"})
end 
if get_config("enable_ing") then 
    sail_add_test("ing", "util", {"SailIng"})
    if get_config("enable_dx") then 
        sail_add_test("ing", "dx", {"SailIng"})
    end
end 
if get_config("enable_cuda") then 
    sail_add_test("cu", "util", {"SailCu"})
end 