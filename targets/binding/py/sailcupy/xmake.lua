target("sailcupy")
    set_languages("cxx20")
    add_deps("SailCu")
    add_deps("external_pybind11")
    add_files("**.cpp")
    add_rules("pybind")

target_end()