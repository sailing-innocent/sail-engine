includes("script/latex.rule.lua")

if (get_config("doc_all")) then 
    local ex_data_path = get_config("ex_data_path")
    includes(ex_data_path)
end

includes("bib")
includes("template")
includes("sample")

-- includes("content")
-- includes("note")
