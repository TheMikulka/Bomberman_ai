import pygame
from .game_state import GameState
from Utilities.button import Button

class PauseState(GameState):
    def __init__(self, game) -> None:
        GameState.__init__(self, game)
        self.resume_image = pygame.image.load('Assets/Buttons/button_resume.png').convert_alpha()
        self.quit_image = pygame.image.load('Assets/Buttons/button_quit.png').convert_alpha()
        
        self.resume_button = Button(self._game.width // 2, self._game.height // 2, self.resume_image, 1)
        self.quit_button = Button(self._game.width // 2, self._game.height // 2 + 100, self.quit_image, 1)
        
    def handle_events(self) -> None:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        if self.resume_button.check_clicked():
            self.exit_state()
        if self.quit_button.check_clicked():
            self.exit_state()
            self._previous_state.exit_state()
        
    def update(self, delta_time) -> None:
        self.handle_events()
        self.resume_button.draw(self._game.screen)
        self.quit_button.draw(self._game.screen)