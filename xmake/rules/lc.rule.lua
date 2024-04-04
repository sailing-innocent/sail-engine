rule("link_lc")
on_load(function(target)
    local lc_path = get_config("lc_path")
    if not lc_path then
        target:set("enabled", false)
        utils.error("lc_path not set.")
        return
    end
    local bin_dir = path.join(lc_path, "bin")
    if is_mode("debug") then
        target:add("linkdirs", path.join(bin_dir, "debug"), {
            public = true
        })
    else
        target:add("linkdirs", path.join(bin_dir, "release"), {
            public = true
        })
    end
    target:add("links", "lc-core", "lc-vstl", "lc-dsl", "lc-gui", "lc-runtime", "lc-ast", "lc-ext-eastl",
        "lc-ext-spdlog", {
            public = true
        })
end)
rule_end()

rule("copy_dll")
after_build(function(target)
    local lc_path = get_config("lc_path")
    if not lc_path then
        return
    end
    local bin_dir
    if is_mode("debug") then
        bin_dir = path.join("bin", "debug")
    else
        bin_dir = path.join("bin", "release")
    end
    local bin_table = {
        "DirectML", "dstorage", "dstoragecore", "dxcompiler", "dxil", 
        "lc-ast", "lc-backend-dx", "lc-core", "lc-ir",
        "lc-ext-eastl", "lc-ext-imgui", "lc-gui", "lc-runtime", "lc-validation-layer",
    }
    local cuda_background_bin_table = {
        -- luisa ext lcub
        "lc-vulkan-swapchain", "lc-backend-cuda",
        "luisa-compute-cuda-ext-dcub", "luisa-compute-cuda-ext-lcub"
    }
    local copy_src_dir = path.join(lc_path, bin_dir)

    for i, v in ipairs(bin_table) do
        os.trycp(path.join(copy_src_dir, v .. ".dll"), bin_dir)
    end

    if get_config("enable_cuda") then
        for i, v in ipairs(cuda_background_bin_table) do
            os.trycp(path.join(copy_src_dir, v .. ".dll"), bin_dir)
        end
        -- for ptx linking
        os.trycp(path.join(copy_src_dir, "luisa_nvrtc.exe"), bin_dir)
    end

    if not is_mode("release") then
        os.cp(path.join(copy_src_dir, "*.pdb"), bin_dir)
    end
end)
rule_end()

target("phony_copy_dll")
set_kind("phony")
add_rules("copy_dll")
before_build(function(target)
    os.mkdir(path.join(os.projectdir(), "bin"))
    os.mkdir(path.join(os.projectdir(), "bin/release"))
    os.mkdir(path.join(os.projectdir(), "bin/debug"))
end)
target_end()

function load_lc()
    local lc_path = get_config("lc_path")
    if lc_path then
        set_values("lc_is_public", true)
        set_values("lc_dir", lc_path)
        add_rules("add_lc_includedirs", "add_lc_defines")
    end
    add_rules("link_lc")
    add_deps("phony_copy_dll")
end
