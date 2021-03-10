from enum import Enum
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
    def __init__(self, header, image_file, story1, story2):
        super(InfoLevel, self).__init__(LevelType.INFO)
        self.header = header
        self.image_file = image_file
        self.story1 = story1
        self.story2 = story2