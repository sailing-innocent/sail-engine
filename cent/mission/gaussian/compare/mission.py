import os 
from mission.base import MissionBase 
from app.project.nvs.reprod_gs.eval import EvalGaussianProjectConfig, EvalGaussianProjectParams, EvalGaussianProject
from module.utils.tex.table import TexTable 

def get_ply_from_json(gs_ply_json, parent_path = ""):
    return os.path.join(parent_path, gs_ply_json['ckpt_path'], "_".join([gs_ply_json['dataset_name'], gs_ply_json['obj_name'], str(gs_ply_json["iter"])]) + ".ply")

class Mission(MissionBase):
    def __init__(self, config_json_file):
        super().__init__(config_json_file, __file__)
        self.name = self.config_json["name"]
        self.render_names = self.config_json["render_names"]
        self.usage = self.config_json["usage"]
        self.scenes = self.config_json["scenes"]
        self.result_template = self.config_json["result_template"]
        self.benchmarks = self.config_json["benchmarks"]
    
    def exec(self):
        proj_config = EvalGaussianProjectConfig(self.env_config)
        proj_config.name = self.name
        proj_config.usage = self.usage
        self.project = EvalGaussianProject(proj_config)

        full_result_tabs = {}
        n_objs = len(self.scenes)
        result_path = self.result_template["path"]
        for benchmark_json in self.benchmarks:
            benchmark = benchmark_json["name"]
            full_result_tab = TexTable(0, n_objs + 1)
            full_result_tab.caption = f"3DGS Profile Results ({benchmark})"
            full_result_tabs[benchmark] = full_result_tab
            # add to template 
            if benchmark_json["use_template"]:
                template = self.result_template["template"]
                template_file_name = os.path.join(result_path, template.replace("{benchmark}", benchmark) + ".json")
                template_tab = TexTable(1, n_objs + 1)
                template_tab.from_json_file(template_file_name)
                full_result_tabs[benchmark].append_rows(template_tab)
        
        for render_name in self.render_names:
            results = self.run_proj(render_name)

            # parser result
            for benchmark in full_result_tabs.keys():
                i = 0
                average = 0
                result_tab = TexTable(1, n_objs + 1)
                # collect the name
                render_name_str = "_"+render_name if len(self.render_names) > 1 else ""
                result_tab.rows[0] = f"3dgs{render_name_str}"
                
                for result in results:
                    result_data = result[benchmark]
                    average += result_data
                    result_tab[0, i] = "{:.2f}".format(float(result_data))
                    result_tab.cols[i] = result["name"]
                    i = i + 1
                
                # average
                average = average / n_objs
                result_tab[0, n_objs] = "{:.2f}".format(average)
                result_tab.cols[n_objs] = "average"
                full_result_tabs[benchmark].append_rows(result_tab)
        
        # EXPORT
        for benchmark in full_result_tabs.keys():    
            result_name = self.result_template["name"].replace("{benchmark}", benchmark)
            target_file_path = os.path.join(result_path, result_name + ".json")
            full_result_tabs[benchmark].to_json_file(target_file_path)

    def run_proj(self, render_name):
        print(f"running scenes on {render_name}")
        for scene in self.scenes:
            assert scene["type"] == "ckpt"
            scene_json = scene.copy()
            ply_file_path = get_ply_from_json(scene, self.env_config.pretrained_path)
            scene_json["ckpt_path"] = ply_file_path
            
            params = EvalGaussianProjectParams(
                scene = scene_json,
                output_name =  scene["dataset_name"] + "_" + scene["obj_name"],
                render_name =  render_name)
            result = self.project.run(params)
            result["name"] = scene["obj_name"]

            yield result

