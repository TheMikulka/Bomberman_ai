class GameState():
    def __init__(self, game) -> None:
        self._game = game
        self._previous_state = None
        
    def update(self, delta_time) -> None:
        pass
    
    def handle_events(self) -> None:
        pass
    
    def enter_state(self) -> None:
        if len(self._game.game_states) > 1:
            self._previous_state = self._game.game_states[-1]
        self._game.game_states.append(self)
    
    def exit_state(self) -> None:
        if len(self._game.game_states) > 1:
            self._game.game_states.pop()
    