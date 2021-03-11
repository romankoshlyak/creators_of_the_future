from enum import Enum
from utils import Images

class LevelType(Enum):
    STUDY_LINE = 1
    INFO = 2
    SPLIT_MONSTERS = 3

class Level(object):
    def __init__(self, level_type):
        self.level_type = level_type

class StudyLineLevel(Level):
    def __init__(self, model, target_model, disabled_buttons):
        super(StudyLineLevel, self).__init__(LevelType.STUDY_LINE)
        self.model = model
        self.target_model = target_model
        self.disabled_buttons = disabled_buttons

class InfoLevel(Level):
    def __init__(self, header, image_file, story):
        super(InfoLevel, self).__init__(LevelType.INFO)
        self.header = header
        self.image = Images.load_image_file(image_file)
        self.story = story

class SplitMonstersLevel(Level):
    def __init__(self, levels, colors, monsters):
        super(SplitMonstersLevel, self).__init__(LevelType.SPLIT_MONSTERS)
        self.levels = levels
        self.colors = colors
        self.monsters = monsters
        self.max_iterations = 3

class MonsterInfo(object):
    def __init__(self, x, y, image_file, target_level):
        self.x = x
        self.y = y
        self.image = Images.load_image(image_file)
        self.target_level = target_level

class SplitMonstersLevelsFactory(object):
    def get_intro_level(self):
        levels = [-100, 0, 100]
        colors = ['blue', 'yellow']
        monsters = [
            MonsterInfo(-2.0, -2.0, Images.BLUE_MONSTER, 0),
            MonsterInfo(-2.0, 2.0, Images.BLUE_MONSTER, 0),
            MonsterInfo(2.0, -2.0, Images.YELLOW_MONSTER, 1),
            MonsterInfo(2.0, 2.0, Images.YELLOW_MONSTER, 1)
        ]
        yield SplitMonstersLevel(levels, colors, monsters) 