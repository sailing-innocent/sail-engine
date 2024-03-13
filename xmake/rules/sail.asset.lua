rule("sail.asset")
    set_extensions(
        ".vert", 
        ".frag",
        ".hlsl",
        ".png",
        ".jpg",
        ".obj"
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
        local glslc = assert(find_tool("glslc"), "glslc not found!")
        -- progress.show(opt.progress, "building glsl %s", sourcefile)
        -- if update
        os.vrunv(glslc.program, {sourcefile, "-o", path.join(targetdir, path.filename(sourcefile) .. ".spv")})
    end)
rule_end()