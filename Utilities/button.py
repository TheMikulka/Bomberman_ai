import pygame
from Utilities.settings import *

class Button():
    def __init__(self, x, y, image: pygame.image, text: str, scale):
        width = image.get_width()
        height = image.get_height()
        self.x = x
        self.y = y
        self.image = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.clicked = False
        self.text = text
        self.font = pygame.font.Font('Assets/Fonts/VCR_OSD_MONO_1.001.ttf', 40)

    def draw_text(self, surface) -> None:
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=(self.x, self.y))
        surface.blit(text_surface, (text_rect.x, text_rect.y))

    def check_clicked(self) -> bool:
        action = False
        position = pygame.mouse.get_pos()

        if self.rect.collidepoint(position):
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
                action = True

        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        return action

    def draw(self, surface):
        surface.blit(self.image, (self.rect.x, self.rect.y))
        self.draw_text(surface)