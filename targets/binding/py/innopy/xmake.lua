target("innopy")
    set_languages("cxx20")
    add_deps("SailInno") -- with its own pybind11
    add_files("**.cpp")
    add_rules("pybind")
target_end()