import torch

from enum import Enum
from utils import Images, Sounds
from models import LinearModel

class LevelType(Enum):
    STUDY_LINE = 1
    INFO = 2
    SPLIT_MONSTERS = 3
    STUDY_PLANE = 4
    MULTI_SPLIT_MONSTERS = 5
    LEARNING_RATE = 6
    LEARNING_RATE_MONSTERS = 7
    DEV_LEVEL = 100

class ErrorType(Enum):
    SUM_LINEAR = 1
    MEAN_LINEAR = 2
    SUM_SQUARED = 3
    MEAN_SQUARED = 4

class PointInfo(object):
    def __init__(self, x, y, target):
        self.x = x
        self.y = y
        self.target = target

class MonsterInfo(PointInfo):
    def __init__(self, x, y, target, image_file):
        super().__init__(x, y, target)
        self.image = Images.load_image(image_file)

class Level(object):
    def __init__(self, level_type):
        self.level_type = level_type
        self.code = None

    def set_level_type(self, level_type):
        self.level_type = level_type

    def set_level_number(self, level_number):
        self.level_number = level_number
        return self

    def set_number_of_levels(self, number_of_levels):
        self.number_of_levels = number_of_levels
        return self

    def set_code(self, code):
        self.code = code
        return self

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

class StudyLineLevel(Level):
    def __init__(self, model, target_model, disabled_buttons, level_number, number_of_levels):
        super().__init__(LevelType.STUDY_LINE, level_number, number_of_levels)
        self.model = model
        self.target_model = target_model
        self.disabled_buttons = disabled_buttons

class StudyPlaneLevel(Level):
    def __init__(self, model, points, step_size, error_type, error_limit):
        super().__init__(LevelType.STUDY_PLANE)
        self.model = model
        self.points = points
        self.step_size = step_size
        self.error_type = error_type
        self.error_limit = error_limit

class LearningRateLevel(StudyPlaneLevel):
    def __init__(self, learning_rate, model, points, step_size, error_type, error_limit):
        super().__init__(model, points, step_size, error_type, error_limit)
        self.learning_rate = learning_rate
        self.set_level_type(LevelType.LEARNING_RATE)

class BaseMonsterLevel(Level):
    def __init__(self, level_type, model, points):
        super().__init__(level_type)
        self.model = model
        self.points = points
        self.step_size = 1.0
        self.max_iterations = 10
        self.hide_restart_button = False
        self.hide_spell = False

    def set_max_iterations(self, max_iterations):
        self.max_iterations = max_iterations
        return self

    def set_hide_restart_button(self, hide_restart_button):
        self.hide_restart_button = hide_restart_button
        return self

    def set_hide_spell(self, hide_spell):
        self.hide_spell = hide_spell
        return self

class SplitMonstersLevel(BaseMonsterLevel):
    def __init__(self, model, points):
        super().__init__(LevelType.SPLIT_MONSTERS, model, points)

class MultiSplitMonstersLevel(BaseMonsterLevel):
    def __init__(self, model, points, step_size, error_type, error_limit):
        super().__init__(LevelType.MULTI_SPLIT_MONSTERS, model, points)
        self.step_size = step_size
        self.error_type = error_type
        self.error_limit = error_limit
        self.learning_rate = 0.1

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

class LearningRateMonstersLevel(MultiSplitMonstersLevel):
    def __init__(self, *args):
        super().__init__(*args)
        self.set_level_type(LevelType.LEARNING_RATE_MONSTERS)

class DevLevel(Level):
    def __init__(self):
        super().__init__(LevelType.DEV_LEVEL)
        self.model = "not implemented"

