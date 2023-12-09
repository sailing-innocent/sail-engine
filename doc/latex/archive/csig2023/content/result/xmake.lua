add_content("tab_csig2023_spatial_hash_vs_bvh")
add_content("tab_csig2023_basic_result_rx7900")

function add_csig_result_content(name)
    target(name)
        add_rules("latex.content")
        add_deps(
            "tab_csig2023_spatial_hash_vs_bvh",
            "tab_csig2023_basic_result_rx7900"
        )
        add_deps("fig_pack_csig2023_result")
        add_files(name .. ".tex")
    target_end()
end

add_csig_result_content("csig2023_result_slide_en")
add_csig_result_content("csig2023_result_doc_cn")
