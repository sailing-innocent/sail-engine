target("SailCodegenPolicy")
    set_kind("headeronly")
    set_group("00.utilities")
    -- dispatch codegen task
    before_build(function(target)
        import("meta.codegen")
        meta_codegen()
    end)

