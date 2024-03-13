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

sail_add_test("basic", "dummy", {"SailDummy"})
if get_config("enable_inno") then 
    sail_add_test("inno", "util", {"SailInno"})
end 
if get_config("enable_ing") then 
    sail_add_test("ing", "util", {"SailIng"})
    sail_add_test("ing", "learn_ogl", {"SailIng", "shaders", "textures"})
end 