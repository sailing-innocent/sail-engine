target("fig_ast_example")
    add_rules("latex.graphviz")
    add_files("ast_example.gv")
target_end()

add_content("abstract_semantic_tree_brief_doc_cn", {
    "fig_ast_example"
})