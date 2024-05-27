package("imgui")
    set_homepage("git@github.com:ocornut/imgui.git")
    set_description("Bloat-free Immediate Mode Graphical User interface for C++ with minimal dependencies")
    set_license("MIT")

    add_versions("2024.05.27-sail", "7203ac3b94460535d610125050127282fdd9653408591ae7dbbee3d5cc364647")

    on_install(function(package)
        os.mkdir(package:installdir())
        os.cd(package:installdir())
        os.cp(path.join(package:scriptdir(), "port", "imgui"), ".")
        os.cp(path.join(package:scriptdir(), "port", "xmake.lua"), "xmake.lua")

        local config = {}
        import("package.tools.xmake").install(package, configs)
    end)
