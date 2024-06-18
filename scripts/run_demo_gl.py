import os 

# WIP: waiting for gl support save image

if __name__ == "__main__":
    out_dir = "D:/workspace/data/result/demo_gl/"
    cmd_prefix = "demo_gl_"
    target_prefix = "demo_gl_"
    targets = [
        "pure"
    ]
    for target in targets:
        cmd = f"{cmd_prefix}{target}"
        outfile = os.path.join(out_dir, f"{target_prefix}{target}.png")
        full_cmd = f"xmake run {cmd} {outfile}"
        print(full_cmd)
        os.system(full_cmd)