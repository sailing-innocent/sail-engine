target("reprod_gs_cuda_util")
    set_kind("static")
    set_languages("c++20")
    set_exceptions("cxx")
    add_deps("external_stb_util")
    add_files("util.cpp")
target_end()

function demo_reprod_gs_cuda(name)
    target("demo_" .. name .. "_reprod_gs_cuda")
    set_kind("binary")
    set_languages("c++20")
    set_exceptions("cxx")
    add_files("demo_" .. name .. ".cpp")
    add_deps("reprod_gs_cuda_util")
end

demo_reprod_gs_cuda("write_image")
