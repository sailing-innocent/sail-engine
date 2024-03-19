rule("latex.graphviz")
    set_extensions(".dot", ".gv")
    add_deps("latex.indirect_content")
    on_load(function (target)
        local ofile = path.join(target:autogendir({root=true}), target:name() .. ".png")
        target:set("values", "targetfile", ofile)
    end)
    on_build_file(function (target, sourcefile, opt)
        import("lib.detect.find_tool")
        import("core.project.depend")
        import("utils.progress")
        depend.on_changed(function()
            local dot = assert(find_tool("dot", {check="-V"}), "dot not found!")
            local ofile = path.join(target:autogendir({root=true}), target:name() .. ".png")
            os.vrunv(dot.program, {"-Tpng", "-o", ofile, sourcefile})
            progress.show(opt.progress, "building graphviz %s", ofile)
        end, {files={sourcefile}})
    end)
rule_end()

rule("latex.ppm_image")
    set_extensions(".ppm")
    add_deps("latex.indirect_content")
    on_load(function (target)
        local ofile = path.join(target:autogendir({root=true}), target:name() .. ".png")
        target:set("values", "targetfile", ofile)
    end)
    on_build_file(function (target, sourcefile, opt)
        import("lib.detect.find_tool")
        import("core.project.depend")
        import("utils.progress")
        -- first build
        local ofile = path.join(target:autogendir({root=true}), target:name() .. ".png")
        depend.on_changed(function()
            local py = assert(find_tool("python", {check="--version"}), "python not found!")
            os.vrunv(py.program, {"doc/latex/script/ppm2png.py", "--ppm", sourcefile, "--png", ofile})
            progress.show(opt.progress, "building ppm %s", ofile)
        end, {files = {sourcefile, ofile} })
    end)
rule_end()

function add_ppm(name)
    target(name)
        add_rules("latex.ppm_image")
        add_files( name .. ".ppm")
    target_end()
end


rule("latex.python_figure")
    set_extensions(".py")
    add_deps("latex.indirect_content")
    on_load(function (target)
        local ofile = path.join(target:autogendir({root=true}), target:name() .. ".png")
        target:set("values", "targetfile", ofile)
    end)
    on_build_file(function (target, sourcefile, opt)
        import("lib.detect.find_tool")
        import("core.project.depend")
        import("utils.progress")
        local ofile = path.join(target:autogendir({root=true}), target:name() .. ".png")
        local py = assert(find_tool("python", {check="--version"}), "python not found!")
        os.vrunv(py.program, {sourcefile, "--target", ofile})
        progress.show(opt.progress, "building pyscript figure %s", ofile)
    end)
    on_link(function (target)
    end)
rule_end()
