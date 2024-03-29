function add_demo(name, deps)
    target(name)
        set_kind("binary")
        add_files(name .. ".cpp")
        set_languages("c++20", "clatest")
        add_deps(deps)
    target_end()
end 

includes("demo")
includes("binding")