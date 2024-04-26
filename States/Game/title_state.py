import pygame
from .game_state import GameState
from .running_state import RunningState
from Utilities.button import Button
from Utilities.asset_loader import AssetLoader

class TitleState(GameState):
    def __init__(self, game) -> None:
        GameState.__init__(self, game)
        self.title_image = pygame.image.load('Assets/Backgrounds/title_screen.png').convert_alpha()
        self.title_image = pygame.transform.scale(self.title_image, (self._game.width, self._game.height))
        self.start_button = Button(self._game.width // 2, self._game.height // 2.2 + 100, AssetLoader().button_img, 'START', 2)
        self.quit_button = Button(self._game.width // 2, self._game.height // 2.2 + 200, AssetLoader().button_img, 'QUIT', 2)

    def handle_events(self) -> None:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        if self.start_button.check_clicked():
            new_state = RunningState(self._game)
            new_state.enter_state()
        if self.quit_button.check_clicked():
            pygame.quit()
            quit()
        
    def update(self, delta_time) -> None:
        self.handle_events()
        self._game.screen.blit(self.title_image, (0, 0))
        self.start_button.draw(self._game.screen)
        self.quit_button.draw(self._game.screen)