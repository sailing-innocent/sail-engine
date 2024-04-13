
target("external_alembic")
set_kind("shared")
add_includedirs("lib", {public = true})
local ale_path = "lib/Alembic/"
add_files(path.join(ale_path, "Abc/*.cpp"), path.join(ale_path, "AbcCollection/*.cpp"),
				path.join(ale_path, "AbcCoreAbstract/*.cpp"), path.join(ale_path, "AbcCoreFactory/*.cpp"),
				path.join(ale_path, "AbcCoreLayer/*.cpp"), path.join(ale_path, "AbcCoreOgawa/*.cpp"),
				path.join(ale_path, "AbcGeom/*.cpp"), path.join(ale_path, "AbcMaterial/*.cpp"), path.join(ale_path, "Ogawa/*.cpp"),
				path.join(ale_path, "Util/*.cpp"))
add_deps("external_imath")
add_defines("ALEMBIC_DLL", {public = true})
add_defines("ABC_WFOBJ_CONVERT_EXPORTS", "ALEMBIC_EXPORTS")
target_end()