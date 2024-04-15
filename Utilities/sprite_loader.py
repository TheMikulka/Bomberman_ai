import pygame
import os
from Utilities.settings import *

def get_sprites(file_path: str, n_frames: int, s_width: int, s_height: int, scale: float) -> list:
        all_sprites = []
        sprite_sheet = pygame.image.load(file_path).convert_alpha()
        for i in range(n_frames):
            image = pygame.Surface((s_width, s_height)).convert_alpha()
            image.blit(sprite_sheet, (0, 0), ((i * s_width), 0, s_width, s_height))
            image = pygame.transform.scale(image, (s_width * scale, s_height * scale))
            image.set_colorkey(BLACK)
            all_sprites.append(image)
            
        return all_sprites

def load(directory: str, n_frames: tuple, s_width: int, s_height: int, scale: float) -> dict:
    # We need to get the right directory in our file system to load all sprites from which is App/Assets/directory
    asset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Assets', directory)

    all_actions_dictionary = {}
    i = 0
    for sub_directory in os.listdir(asset_directory):
        sub_directory_path = os.path.join(asset_directory, sub_directory)
        # Check if sub_directory is a directory and is not empty
        if os.path.isdir(sub_directory_path) and any(os.path.isfile(os.path.join(sub_directory_path, f)) for f in os.listdir(sub_directory_path)):
            all_actions_dictionary[sub_directory] = {}
            for image in os.listdir(sub_directory_path):
                current_image_path = os.path.join(sub_directory_path, image)
                all_actions_dictionary[sub_directory][image.split('.')[0]] = get_sprites(current_image_path, n_frames[i], s_width, s_height, scale)
        else:
            all_actions_dictionary[sub_directory.split('.')[0]] = get_sprites(sub_directory_path, n_frames[i], s_width, s_height, scale)
            
        i += 1
        
    return all_actions_dictionary