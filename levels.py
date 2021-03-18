import torch

from enum import Enum
from utils import Images, Sounds
from models import LinearModel

class LevelType(Enum):
    STUDY_LINE = 1
    INFO = 2
    SPLIT_MONSTERS = 3
    STUDY_PLANE = 4

class Level(object):
    def __init__(self, level_type, level_number = 1, number_of_levels = 1):
        self.level_type = level_type
        self.level_number = level_number
        self.number_of_levels = number_of_levels

class StudyLineLevel(Level):
    def __init__(self, model, target_model, disabled_buttons, level_number, number_of_levels):
        super().__init__(LevelType.STUDY_LINE, level_number, number_of_levels)
        self.model = model
        self.target_model = target_model
        self.disabled_buttons = disabled_buttons


class ErrorType(Enum):
    SUM_LINEAR = 1
    MEAN_LINEAR = 2
    SUM_SQUARED = 3
    MEAN_SQUARED = 4

class StudyPlaneLevel(Level):
    def __init__(self, model, points, error_type, error_limit, level_number, number_of_levels):
        super().__init__(LevelType.STUDY_PLANE, level_number, number_of_levels)
        self.model = model
        self.points = points
        self.error_type = error_type
        self.error_limit = error_limit

class PointInfo(object):
    def __init__(self, x, y, target):
        self.x = x
        self.y = y
        self.target = target

class StudyPlaneLevelFactory(object):
    def get_tensor(self, points):
        return torch.tensor(points, dtype=torch.float)

    def get_points(self, target_model, points):
        output = target_model(self.get_tensor(points)).view(-1).tolist()
        return [PointInfo(x, y, z) for (x, y), z in zip(points, output)]

    def get_study_levels(self):
        model = LinearModel(0.5, 0.5, 0.5)
        target_model = LinearModel(0.2, 0.1, -0.5)
        points = [(1, 2), (-2, 0)]
        yield StudyPlaneLevel(model, self.get_points(target_model, points), ErrorType.SUM_LINEAR, 0.5, 1, 3)
        model = LinearModel(0.4, 0.4, 0.2)
        target_model = LinearModel(-0.3, -0.5, -0.4)
        points = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        yield StudyPlaneLevel(model, self.get_points(target_model, points), ErrorType.SUM_LINEAR, 0.5, 2, 3)
        model = LinearModel(0.0, 0.0, 0.0)
        target_model = LinearModel(1.0, 0.5, -1.0)
        points = [(1, 4), (5, 3), (4, -2), (-4, 6), (3, -2)]
        yield StudyPlaneLevel(model, self.get_points(target_model, points), ErrorType.SUM_LINEAR, 2.0, 3, 3)

class InfoLevel(Level):
    def __init__(self, header, image_file, audio_file, story = None):
        super().__init__(LevelType.INFO)
        self.header = header
        self.image = Images.load_image_file(image_file)
        self.audio_file = Sounds.get_file(audio_file)
        self.story = story
        self.hide_next_button = False

    def set_hide_next_button(self, hide_next_button):
        self.hide_next_button = hide_next_button
        return self

class SplitMonstersLevel(Level):
    def __init__(self, model, levels, colors, monsters, level_number, number_of_levels):
        super().__init__(LevelType.SPLIT_MONSTERS, level_number, number_of_levels)
        self.model = model
        self.levels = levels
        self.colors = colors
        self.monsters = monsters
        self.level_number = level_number
        self.number_of_levels = number_of_levels
        self.max_iterations = 10
        self.hide_restart_button = False

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations
        return self

    def set_hide_restart_button(self, hide_restart_button):
        self.hide_restart_button = hide_restart_button
        return self

class MonsterInfo(object):
    def __init__(self, x, y, image_file, target_level):
        self.x = x
        self.y = y
        self.image = Images.load_image(image_file)
        self.target_level = target_level

class SplitMonstersLevelsFactory(object):
    def get_intro_level(self):
        model = LinearModel(0.5, 0.0, 0.0)
        levels = [-100, 0, 100]
        colors = ['#0f90bf', '#dbd1ed']
        monsters = [
            MonsterInfo(-2.0, -2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(0.0, -2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(2.0, -2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(-2.0, 2.0, Images.SNOW_MONSTER, 1),
            MonsterInfo(0.0, 2.0, Images.SNOW_MONSTER, 1),
            MonsterInfo(2.0, 2.0, Images.SNOW_MONSTER, 1)
        ]
        yield SplitMonstersLevel(model, levels, colors, monsters, 1, 1).set_max_iterations(3).set_hide_restart_button(True)

    def get_main_levels(self):
        model = LinearModel(0.5, 0.0, 0.0)
        levels = [-100, 0, 100]
        colors = ['#0f90bf', '#dbd1ed']
        monsters = [
            MonsterInfo(-2.0, -2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(0.0, -2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(2.0, -2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(-2.0, 2.0, Images.SNOW_MONSTER, 1),
            MonsterInfo(0.0, 2.0, Images.SNOW_MONSTER, 1),
            MonsterInfo(2.0, 2.0, Images.SNOW_MONSTER, 1)
        ]
        yield SplitMonstersLevel(model, levels, colors, monsters, 1, 3)

        model = LinearModel(0.5, 0.0, 0.0)
        levels = [-100, 0, 100]
        colors = ['#0f90bf', '#dbd1ed']
        monsters = [
            MonsterInfo(-2.0, 2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(0.0, 2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(2.0, 2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(-2.0, 1.0, Images.SNOW_MONSTER, 1),
            MonsterInfo(0.0, 1.0, Images.SNOW_MONSTER, 1),
            MonsterInfo(2.0, 1.0, Images.SNOW_MONSTER, 1)
        ]
        yield SplitMonstersLevel(model, levels, colors, monsters, 2, 3).set_max_iterations(30)

        model = LinearModel(-0.5, 0.5, 1.0)
        levels = [-100, 0, 100]
        colors = ['#0f90bf', '#dbd1ed']
        monsters = [
            MonsterInfo(1.0, 2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(2.0, 1.0, Images.DINO_MONSTER, 0),
            MonsterInfo(-2.0, -2.0, Images.DINO_MONSTER, 0),
            MonsterInfo(2.0, -2.0, Images.SNOW_MONSTER, 1),
        ]
        yield SplitMonstersLevel(model, levels, colors, monsters, 3, 3).set_max_iterations(30)
