target("common_bib")
    add_rules("latex.common_bib")
    add_files("common.bib")
target_end()

if (get_config("doc_all")) then 
    add_bib("sample", {})
end 
