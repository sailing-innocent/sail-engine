rule("sail.asset")
    set_extensions(
        ".vert", 
        ".frag",
        ".hlsl",
        ".png",
        ".jpg",
        ".obj",
        ".ply",
        ".gltf",
        ".bin"
    )
    on_build_file(function(target, sourcefile, opt) 
        -- if not exist directory, create it
        local targetdir = path.join(target:targetdir(), path.directory(sourcefile))
        if not os.isdir(targetdir) then
            os.mkdir(targetdir)
        end
        os.cp(sourcefile, path.join(targetdir, path.filename(sourcefile)))
    end)
rule_end()

rule("sail.glsl")
    set_extensions(
        ".vert",
        ".frag"
    )
    on_build_file(function(target, sourcefile, opt)
        local targetdir = path.join(target:targetdir(), path.directory(sourcefile))
        if not os.isdir(targetdir) then
            os.mkdir(targetdir)
        end
        import("lib.detect.find_tool")
        import("core.project.depend")
        import("utils.progress")
        local ofile = path.join(targetdir, path.filename(sourcefile) .. ".spv")
        depend.on_changed(function()
            local glslc = assert(find_tool("glslc"), "glslc not found!")
            progress.show(opt.progress, "building glsl %s", sourcefile)
            os.vrunv(glslc.program, {sourcefile, "-o", ofile})
        end, { files = {sourcefile, ofile}})
    end)
rule_end()