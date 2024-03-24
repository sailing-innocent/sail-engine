-- reflection system

function sail_module_gen_json(target, filename, dep_modules) 
    local last = os.time()
    -- if not depend not changed, early return
    -- depend file
    -- depend info
    -- if depend not changed, return

    local pub_deps = target:values("sail.module.public_dependencies")
    -- start rebuild json
    local json_content = "{\n\"module\":{\n"
    json_content = json_content .. "\"name\":\"" .. target:name() .. "\",\n"
    -- dep body
    json_content = json_content .. "\"deps\":[\n "

end