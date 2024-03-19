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

rule("pybind")

on_load(function (target)
    target:set("kind", "shared")
    target:set("extension", ".pyd")
    function string_split(str, chr)
        local map = {}
        for part in string.gmatch(str, "([^" .. chr .. "]+)") do
            table.insert(map, part)
        end
        return map
    end
    local function find_process_path(process)
        local cut
        local is_win = os.is_host("windows")
        if is_win then
            cut = ";"
        else
            cut = ":"
        end
        local path_str = os.getenv("PATH")
        if path_str then
            local paths = string_split(path_str, cut)
            for i, pth in ipairs(paths) do
                if os.isfile(path.join(pth, process)) then
                    return pth
                end
            end
        end
        return nil
    end
    local py_path = find_process_path("python.exe")
    local py_include = nil
    local py_linkdir = nil
    local py_libs = nil

    if py_path then
        py_include = path.join(py_path, "include")
        py_linkdir = path.join(py_path, "libs")
        local files = {}
        for _, filepath in ipairs(os.files(path.join(py_linkdir, "*.lib"))) do
            local lib_name = path.basename(filepath)
            table.insert(files, lib_name)
        end
        py_libs = files
    end
    target:add("includedirs", py_include)
    target:add("linkdirs", py_linkdir)
    target:add("syslinks", py_libs)
end)
rule_end()