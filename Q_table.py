import json

class Q_table:
    def __init__(self,filename, defaut_len):
        self.filename = filename
        self.table, self.epsilon = self.load()
        self.default_len = defaut_len

    def get(self, key:str) -> list:
        if key in self.table:
            return self.table[key]
        else:
            return [0 for i in range(self.default_len)]
        
    def set(self, key:str, data:list):
        if sum(data) == 0:
            return
        self.table[key] = data
    
    def save(self):
        with open(self.filename, "w") as f:
            json.dump({"Epsilon":self.epsilon,"Q_table":self.table}, f)

    def load(self):
        try:
            with open(self.filename, "r") as f:
                data = json.load(f)
                epsilon = data["Epsilon"]
                Q_table = data["Q_table"]
        except FileNotFoundError:
            Q_table = {}
            epsilon = 1.0
            with open(self.filename, "w") as f:
                json.dump({"Epsilon":epsilon,"Q_table":Q_table}, f)
        return Q_table, epsilon