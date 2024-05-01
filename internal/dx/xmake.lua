SHARED_MODULE("SailDX", "SAIL_DX", engine_version)
    add_includedirs("include", {public=true})
    add_files("src/**.cpp")
    add_links("d3d12", "dxgi", "D3DCompiler", {public = true})