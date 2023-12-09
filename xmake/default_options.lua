-- lc and inno
option("lc_path")
    set_default(false)
    set_showmenu(true)
option_end()
option("enable_inno")
    set_default(true)
    set_showmenu(true)
option_end()

-- ing 
option("enable_ing")
    set_default(true)
    set_showmenu(true)
option_end()

-- gl_support
option("enable_gl")
    set_default(true)
    set_showmenu(true)
option_end()

-- vk_support
option("enable_vk")
    set_default(true)
    set_showmenu(true)
option_end()

-- cuda_support
option("enable_cuda")
    set_default(true)
    set_showmenu(true)
option_end()

-- dx_support
option("enable_dx")
    set_default(true)
    set_showmenu(true)
option_end()

-- doc
option("enable_doc")
    set_default(true)
    set_showmenu(true)
option_end()

option("latex_out")
set_default("D:/workspace/doc")
set_showmenu(true)
option_end()

option("doc_all")
set_default(false)
set_showmenu(true)
set_description("with all document")
option_end()

option("ex_data_path")
set_default("D:/workspace/data")
set_showmenu(true)
set_description("the path of the external data")
option_end()
