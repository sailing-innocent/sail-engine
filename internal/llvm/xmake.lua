SHARED_MODULE("SailLLVM", "SAIL_LLVM", engine_version)
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_deps("SailBase")
    local llvm_path = get_config("llvm_path")
    add_includedirs(llvm_path .. "/include", {public=true})
    add_linkdirs(llvm_path .. "/lib", {public=true})
    set_pcxxheader("src/pch.h")
    on_load(function(target, opt)
        local libs = {}
        local llvm_path = get_config("llvm_path")
        local p = llvm_path .. "/lib/*.lib"
        for __, filepath in ipairs(os.files(p)) do
            local basename = path.basename(filepath)
            table.insert(libs, basename)
        end
        target:add("links", libs, {public=true})
    end)