import pygame
from .game_state import GameState
from .running_state import RunningState

class TitleState(GameState):
    def __init__(self, game) -> None:
        GameState.__init__(self, game)
        
    def handle_events(self) -> None:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_SPACE:
                        new_state = RunningState(self._game)
                        new_state.enter_state()
        
    def update(self, delta_time) -> None:
        self.handle_events()
        self._game.screen.fill((255, 255, 255))
        self._game.draw_text('Bomberman', self._game.width // 2, self._game.height // 2)