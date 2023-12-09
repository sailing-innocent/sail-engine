rule("sail.cuda")
    on_load(function(target)
        local cuda_path = os.getenv("CUDA_PATH")
        if cuda_path then
            target:add("sysincludedirs", path.join(cuda_path, "include"), {public=true})
            target:add("linkdirs", path.join(cuda_path, "lib/x64/"), {public=true})
            target:add("links", "nvrtc", "cudart", "cuda", {public=true})
        else
            target:set("enabled", false)
            return
        end
        if is_plat("windows") then
            target:add("defines", "NOMINMAX", "UNICODE")
            target:add("syslinks", "Cfgmgr32", "Advapi32")
        end
    end)

    after_build(function(target)
        local cuda_path = os.getenv("CUDA_PATH")
        if cuda_path then
            os.cp(path.join(cuda_path, "bin/*.dll"), target:targetdir())
            -- for indirect dispatch
            -- os.cp(path.join(cuda_path, "lib/x64/cudadevrt.lib"), target:targetdir())
        end
    end)
rule_end()

rule("lc.lcub")
    on_load(function(target)
        target:add("links", "luisa-compute-cuda-ext-lcub", "luisa-compute-cuda-ext-dcub", {public = true})
    end)
rule_end()