includes("script/latex.rule.lua")
includes("script/table.rule.lua")
includes("script/figure.rule.lua")

if (get_config("doc_all")) then 
end

local ex_data_path = get_config("ex_data_path")
includes(ex_data_path)

includes("bib")
includes("template")
includes("sample")
includes("figure")
includes("table")
includes("content")
includes("note")
includes("archive")
includes("publish")

includes("pbd_gs")