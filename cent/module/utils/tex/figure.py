from module.utils.image.basic import Image 
from typing import NamedTuple
import os 

class TexFigData:
    def __init__(self):
        self.data = []

    def export(self, filepath=None):
        content = "\n".join([" ".join([str(dt) for dt in data_item]) for data_item in self.data])
        if filepath is not None:
            with open(filepath, 'w') as tf:
                tf.write(content)
        else:
            return content

class LineData(NamedTuple):
    data: TexFigData
    file_name: str
    titlef: str 

class TexFigure:
    def __init__(self):
        self.lines = []
        self.title = "dummy"
        self.x_label = "x"
        self.y_label = "y"
        self.legend_style = {
            "at": [0.97, 0.5],
            "anchor": "west"
        }
        self.legends = []

    def add_line(self, data: TexFigData, title: str, prefix: str = "result", postfix: str = "data"):
        f = title.replace(" ","_")
        file_name = f"{prefix}_{f}_{postfix}"
        dt = LineData(data, file_name, f)
        self.lines.append(dt)
        self.legends.append(title)

    def export(self, target_dir):
        print("exporting to {}".format(target_dir))
        for line in self.lines:
            line.data.export(os.path.join(target_dir, f"{line.file_name}.dat"))
        t = self.title.lower().replace(" ", "_")
        tt = f"pgf_{t}_result"
        pgf_file = os.path.join(target_dir, tt + ".tex")

        print(f"exporting to {pgf_file}")
        legend_style_content = f"at={{({self.legend_style['at'][0]},{self.legend_style['at'][1]})}},anchor={self.legend_style['anchor']}"
        legend_entries_content = ",\n".join([f"{legend}" for legend in self.legends])
        add_plot_content = "\n".join([f"\\addplot table {{{line.file_name}.dat}};" for line in self.lines])
        pgf_content = f"""
\\begin{{tikzpicture}}
\t\\begin{{axis}}[
\t\t title={{{self.title}}},
\t\t xlabel={{{self.x_label}}},
\t\t ylabel={{{self.y_label}}},
\t\t legend style={{{legend_style_content}}},
\t\t legend entries={{\n{legend_entries_content}\n }},
\t\t]
{add_plot_content}
\t\\end{{axis}}
\\end{{tikzpicture}}
        """
        # print(pgf_content)
        with open(pgf_file, "w") as f:
            f.write(pgf_content)
        xmake_file = os.path.join(target_dir, "xmake.lua")
        dat_content = "\n".join([f"add_dat(\"{line.file_name}\")" for line in self.lines])
        deps_content = "{" + ",\n".join([f"\"{line.file_name}\"" for line in self.lines]) + "}"
        xmake_content = f"""
{dat_content}
add_content(\"{tt}\",\n{deps_content}\n)\n
"""
        with open(xmake_file, "w") as f:
            f.write(xmake_content)

class ImageFigure:
    def __init__(self, img: Image):
        self.img = img 

    def export(self, filepath=None):
        pass 
        # if filepath exists, write to file path
        # else, return the content ( name, build_str )

class FigPack:
    def __init__(self):
        self.figs = []
        self.name = "fig_pack_dummy"

    def export(self, filepath=None):
        pass 
        # if filepath exists, write to file path
        # else, return the content ( name, build_str )