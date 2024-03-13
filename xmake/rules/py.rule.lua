rule("py_source")
    set_extensions(".py")
    on_load(function(target, opt)
        target:set("kind", "object")
    end)
    on_build_file(function (target, sourcefile, opt)
        -- import("utils.progress")
        os.cp(sourcefile, target:targetdir())
        -- progress.show(opt.progress, "moving %s", sourcefile)
    end)
    on_link(function(target, opt)
        -- nothing to do
    end)
rule_end()

rule("py_dir")
    add_deps("py_source")
    on_load(function(target, opt)
        local targetdir = path.join(target:targetdir(), target:name())
        os.mkdir(targetdir)
        target:set("targetdir", targetdir)
    end)
    after_build(function(target, opt)
        -- generate __init__.py with debug/release info
        local init_py = path.join(target:targetdir(), "__init__.py")
        local f = io.open(init_py, "w")
        f:write("import sys\n")
        local bin_path = "bin/debug"
        if is_mode("release") then
            bin_path = "bin/release"
        end
        f:write("sys.path.append('" .. bin_path .. "')\n")
        f:close()
    end)
rule_end()
