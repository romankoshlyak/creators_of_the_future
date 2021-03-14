from enum import Enum
from utils import Images, Sounds
from models import LinearModel

class LevelType(Enum):
    STUDY_LINE = 1
    INFO = 2
    SPLIT_MONSTERS = 3

class Level(object):
    def __init__(self, level_type):
        self.level_type = level_type

class StudyLineLevel(Level):
    def __init__(self, model, target_model, disabled_buttons, level_number, number_of_levels):
        super(StudyLineLevel, self).__init__(LevelType.STUDY_LINE)
        self.model = model
        self.target_model = target_model
        self.disabled_buttons = disabled_buttons
        self.level_number = level_number
        self.number_of_levels = number_of_levels

class InfoLevel(Level):
    def __init__(self, header, image_file, audio_file, story = None):
        super(InfoLevel, self).__init__(LevelType.INFO)
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
        super(SplitMonstersLevel, self).__init__(LevelType.SPLIT_MONSTERS)
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
