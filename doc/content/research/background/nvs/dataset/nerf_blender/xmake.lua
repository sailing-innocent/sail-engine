add_content("fig_dataset_nerf_blender_chair")
add_content("fig_dataset_nerf_blender_drums")
add_content("fig_dataset_nerf_blender_ficus")
add_content("fig_dataset_nerf_blender_hotdog")
add_content("fig_dataset_nerf_blender_lego")
add_content("fig_dataset_nerf_blender_material")
add_content("fig_dataset_nerf_blender_mic")
add_content("fig_dataset_nerf_blender_ship")

target("fig_pack_nerf_blender")
    set_kind("phony")
    add_deps(
        "fig_pack_demo_nerf_blender",
        "fig_dataset_nerf_blender_chair",
        "fig_dataset_nerf_blender_drums",
        "fig_dataset_nerf_blender_ficus",
        "fig_dataset_nerf_blender_hotdog",
        "fig_dataset_nerf_blender_lego",
        "fig_dataset_nerf_blender_material",
        "fig_dataset_nerf_blender_mic",
        "fig_dataset_nerf_blender_ship")

target_end()


add_content("dataset_nerf_blender_intro_doc_en", {
    "fig_pack_nerf_blender"
})
add_content("dataset_nerf_blender_intro_doc_cn", {
    "fig_pack_nerf_blender"
})
add_content("dataset_nerf_blender_intro_slide_cn",{
    "fig_pack_nerf_blender"
})
