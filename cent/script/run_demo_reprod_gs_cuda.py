import os 

if __name__ == "__main__":
    out_dir = "D:/workspace/data/result/demo_reprod_gs_cuda/"
    cmd_prefix = "demo_rgcu_"
    target_prefix = "demo_reprod_gs_cuda_"
    targets = [
        "debug_img"
    ]
    for target in targets:
        cmd = f"{cmd_prefix}{target}"
        outfile = os.path.join(out_dir, f"{target_prefix}{target}.png")
        full_cmd = f"xmake run {cmd} {outfile}"
        print(full_cmd)
        os.system(full_cmd)