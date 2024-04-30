
class Monitoring():
    __instance = None
    
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Monitoring, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        if Monitoring.__instance is None:
            Monitoring.__instance = self
            self.win_counter = {}
            print("Monitoring created")
    
    def add_win(self, player: str) -> None:
        if player in self.win_counter:
            self.win_counter[player] += 1
        else:
            self.win_counter[player] = 1