from abc import ABC, abstractmethod

class TexResult(ABC):
    def __init__(self):
        self.name = "tex_result_dummy"
        pass 

    @abstractmethod
    def export_impl(self):
        pass 

    def export(self, filename=None):
        content = self.export_impl()
        if filename is None:
            return content
        else:
            with open(filename, "w") as f:
                f.write(content)

class ResultDir(TexResult):
    def __init__(self):
        self.name = "result_dir_dummy"

    def export_impl(self):
        return ""