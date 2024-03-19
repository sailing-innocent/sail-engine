import json 
# Col-Major
class TexTable:
    def __init__(self, n_rows: int = 0, n_cols: int = 0, caption: str = "", label: str = ""):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self._rows = ["" for i in range(n_rows)] # first col
        self._cols = ["" for i in range(n_cols)] # first row
        
        self._data = [["" for i in range(n_cols)] for j in range(n_rows)]
        self.caption = caption
        self.label = label

    def from_json(self, data_json):
        self._cols = data_json["cols"]
        self._rows = data_json["rows"]
        self._data = data_json["data"]
        self.caption = data_json["caption"]
        self.label = data_json["label"]

    def from_json_file(self, json_file: str = ""):
        with open(json_file, 'r') as jf:
            data_json = json.loads(jf.read())
            jf.close()
        self.from_json(data_json)
    
    def to_json(self):
        data_json = {
            "rows": self._rows,
            "cols": self._cols,
            "data": self._data,
            "caption": self.caption,
            "label": self.label
        }
        return data_json 
    
    def to_json_file(self, file_path: str = ""):
        data_json = self.to_json()
        with open(file_path, 'w') as tf:
            json.dump(data_json, tf)
            tf.close()

    def format(self):
        # i, j: row_id, col_id
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self._data[i][j] = "{:0.2f}".format(self._data[i][j])

    @property
    def rows(self):
        return self._rows
    
    @rows.setter
    def rows(self, value):
        self._rows = value

    @property
    def cols(self):
        return self._cols
    
    @cols.setter
    def cols(self, value):
        self._cols = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self.n_rows = len(value)
        self.n_cols = len(value[0])
        self._data = value

    @property
    def caption(self):
        return self._caption
    
    @caption.setter
    def caption(self, value):
        self._caption = value

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, value):
        self._label = value

    def __getitem__(self, idx):
        i, j = idx 
        return self._data[i][j]

    def __setitem__(self, idx, value):
        i, j = idx 
        self._data[i][j] = value

    def insert_col(self, i):
        self.n_cols = self.n_cols + 1
        self._cols.insert(i, "")
        for i in range(self.n_rows):
            self._data[i].insert(i, "")

    def append_rows(self, tab):
        # it's user's responsibility to make sure the col dimension is correct
        if self.n_rows == 0:
            # if first row is empty
            self.cols = tab.cols
        self.n_rows = self.n_rows + tab.n_rows 
        self.rows = self.rows + tab.rows
        self._data = self._data + tab.data