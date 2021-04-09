import hashlib
from utils import Images
from models import LinearModel
from levels import *

class BaseLevelFactory(object):
    def get_tensor(self, points):
        return torch.tensor(points, dtype=torch.float)

    def get_outputs(self, model, points):
        return model(self.get_tensor(points)).view(-1).tolist()

    def check_outputs(self, outputs, step_size):
        for i, output in enumerate(outputs):
            if (output+1e-4)%step_size > 1e-3:
                print(i, output)
                assert(False)

class StudyPlaneLevelFactory(BaseLevelFactory):
    def get_points(self, target_model, points, step_size):
        outputs = self.get_outputs(target_model, points)
        self.check_outputs(outputs, step_size)
        return [PointInfo(x, y, z) for (x, y), z in zip(points, outputs)]

    def get_learning_rate_levels(self):
        model = LinearModel(-0.5, 0.5, 0.0, False)
        target_model = LinearModel(0.33, -0.33, 0.0, False)
        points = [(-2.0, -2.0), (-2.0, 1.0), (-2.0, 2.0), (1.0, 2.0), (2.0, 0.0)]
        step_size = 0.33
        points = self.get_points(target_model, points, step_size)
        learning_rate = 1.0
        yield LearningRateLevel(learning_rate, model, points, step_size, ErrorType.SUM_LINEAR, 0.0001)

        model = LinearModel(0.4, 0.4, 0.2)
        target_model = LinearModel(-0.25, -0.5, 0.0)
        points = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        step_size = 0.5
        points = self.get_points(target_model, points, step_size)
        learning_rate = 0.5
        yield LearningRateLevel(learning_rate, model, points, step_size, ErrorType.SUM_LINEAR, 0.5)

        model = LinearModel(-0.5, 0.5, 0.5)
        target_model = LinearModel(0.33, -0.33, 0.0)
        points = [(-2.0, -2.0), (-2.0, 1.0), (-2.0, 2.0), (1.0, 2.0), (2.0, 0.0)]
        step_size = 0.33
        points = self.get_points(target_model, points, step_size)
        learning_rate = 1.0
        yield LearningRateLevel(learning_rate, model, points, step_size, ErrorType.SUM_LINEAR, 0.0001)

    def get_study_levels(self):
        model = LinearModel(0.1, 0.1, 0.5)
        target_model = LinearModel(0.5, 0.5, -0.5)
        points = [(1, 2), (-2, 1)]
        step_size = 1.0
        points = self.get_points(target_model, points, step_size)
        yield StudyPlaneLevel(model, points, step_size, ErrorType.SUM_LINEAR, 0.5, 1, 3)
        model = LinearModel(0.4, 0.4, 0.2)
        target_model = LinearModel(-0.25, -0.5, 0.0)
        points = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        step_size = 0.5
        points = self.get_points(target_model, points, step_size)
        yield StudyPlaneLevel(model, points, step_size, ErrorType.SUM_LINEAR, 0.5, 2, 3)
        model = LinearModel(-0.5, -0.5, 0.5)
        target_model = LinearModel(1.0, 0.5, -1.0)
        points = [(1, 2), (1, 0), (2, -2), (-2, 2), (1, -2), (-2, -2)]
        step_size = 1.0
        points = self.get_points(target_model, points, step_size)
        yield StudyPlaneLevel(model, points, step_size, ErrorType.SUM_LINEAR, 2.0, 3, 3)

class MonstersLevelsFactory(BaseLevelFactory):
    def get_monsters(self, target_model, points, step_size):
        outputs = self.get_outputs(target_model, points)
        self.check_outputs(outputs, step_size)
        return [MonsterInfo(x, y, z, Images.DINO_MONSTER) for (x, y), z in zip(points, outputs)]

    def get_learning_rate_levels(self):
        model = LinearModel(-0.5, 0.50, 0.50)
        target_model = LinearModel(0.5, -0.5, 0)
        points = [(-2.0, -2.0), (-2.0, 1.0), (-2.0, 2.0), (1.0, 2.0), (2.0, 0.0)]
        step_size = 0.5
        points = self.get_monsters(target_model, points, step_size)
        yield LearningRateMonstersLevel(model, points, step_size, ErrorType.SUM_LINEAR, 0.5).set_max_iterations(30).set_learning_rate(1.0)
        model = LinearModel(-0.2, 0.6, 0.4)
        target_model = LinearModel(0.2, -0.6, -0.4)
        points = [(-2.0, -2.0), (-2.0, 1.0), (-2.0, 2.0), (1.0, 2.0), (2.0, 0.0)]
        step_size = 0.2
        points = self.get_monsters(target_model, points, step_size)
        yield LearningRateMonstersLevel(model, points, step_size, ErrorType.SUM_LINEAR, 0.5).set_max_iterations(30).set_hide_spell(True).set_learning_rate(1.0)
        model = LinearModel(-0.22, 0.64, 0.12)
        target_model = LinearModel(0.34, -0.23, -0.4)
        points = []
        for x in range(-2, 3):
            for y in range(-2, 3):
                points.append((x, y))
        step_size = 0.01
        points = self.get_monsters(target_model, points, step_size)
        yield LearningRateMonstersLevel(model, points, step_size, ErrorType.SUM_LINEAR, 0.5).set_max_iterations(30)

    def get_multi_split_levels(self):
        model = LinearModel(0.0, 0.5, 0.2)
        target_model = LinearModel(1.0, 0.5, -0.5)
        points = [(-2.0, -1.0), (0.0, 1.0), (2.0, 1.0)]
        step_size = 1.0
        points = self.get_monsters(target_model, points, step_size)
        yield MultiSplitMonstersLevel(model, points, step_size, ErrorType.SUM_LINEAR, 0.5, 1, 3).set_max_iterations(30)
        model = LinearModel(-0.5, 0.5, 0.5)
        target_model = LinearModel(0.3, -0.3, 0.0)
        points = []
        for x in range(-2, 3):
            for y in range(-2, 3):
                points.append((x, y))
        step_size = 0.3
        points = self.get_monsters(target_model, points, step_size)
        yield MultiSplitMonstersLevel(model, points, step_size, ErrorType.SUM_LINEAR, 0.5, 2, 3).set_max_iterations(60).set_hide_spell(True)
        model = LinearModel(-0.2, 0.6, 0.4)
        target_model = LinearModel(0.2, -0.6, -0.4)
        points = [(-2.0, -2.0), (-2.0, 1.0), (-2.0, 2.0), (1.0, 2.0), (2.0, 0.0)]
        step_size = 0.2
        points = self.get_monsters(target_model, points, step_size)
        yield MultiSplitMonstersLevel(model, points, step_size, ErrorType.SUM_LINEAR, 0.5, 3, 3).set_max_iterations(60)

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

class MainLevelsFactory(BaseLevelFactory):
    def set_level_codes(self, salt, gen):
        levels = list(gen)
        for i, level in enumerate(levels):
            sha = hashlib.sha256()
            sha.update(f'{salt}:{i}'.encode())
            code = sha.hexdigest()[:6].strip().lower()
            yield level.set_code(code)

    def set_level_numbers(self, gen):
        levels = list(gen)
        for i, level in enumerate(levels):
            yield level.set_level_number(i+1).set_number_of_levels(len(levels))

    def all_levels(self):
        #yield from self.first_level()
        #yield from self.second_level()
        yield from self.set_level_codes("SALT_3", self.third_level())

    def third_level(self):
        yield InfoLevel("Let me prepare myself for next night", "./images/wake_up.jpg", None, "It was easy after preparation, let's prepare today too")
        yield from self.set_level_numbers(StudyPlaneLevelFactory().get_learning_rate_levels())
        yield InfoLevel("You are back", "./images/dream.jpg", None, "There are new <b>Lernos Ratos</b> spells for you:</br><b>Minisimus</b> - set lernos ratos to minimum value, you can cast <b>Iteratimus</b> spells for free</br><b>Restorisimus</b> - restore lernos ratos to previous value</br><b>Incrisimus/Dicrisimus</b> - increase/decrease lernos ratos, note you can not use it till you call Restorisimus after Minisimus")
        yield from self.set_level_numbers(MonstersLevelsFactory().get_learning_rate_levels())
        yield InfoLevel("Congratulations", "./images/dream.jpg", None, "Congratulations! You once again show your potential, the way you handlered hide spell was impressive").set_hide_next_button(True)


    def second_level(self):
        yield InfoLevel("Let me prepare myself for next night", "./images/wake_up.jpg", None, "It was easy after preparation, let's prepare today too")
        yield from StudyPlaneLevelFactory().get_study_levels()
        yield InfoLevel("You are back", "./images/dream.jpg", None, "Warning, <b>Iterasimums</b> learned a new hide spell and he can hide battle field from you, but <b>Acurasimus</b> bring his friend <b>Erorisimus</b> to help out in such situation")
        yield from SplitMonstersLevelsFactory().get_multi_split_levels()
        yield InfoLevel("Congratulations", "./images/dream.jpg", None, "Congratulations! You once again show your potential, the way you handlered hide spell was impressive").set_hide_next_button(True)

    def first_level(self):
        yield from self.intro_levels()
        yield from self.study_line_levels()
        yield from self.monster_levels()

    def intro_levels(self):
        yield InfoLevel("After a long day", "./images/sleep.jpg", "after_a_long_day", "After a long day, it's time to go to sleep<br/>Click Next level, to continue...")
        yield InfoLevel("Welcome", "./images/dream.jpg", "welcome", "Welcome to the creators world. You have been choosen to fight on the side of the future. There is no time to get a proper training, since we are in the middle of the battle, so I will set up magic interface for you. Just cast spells and we hope you will lead <b>Acurasimus</b> to the victory over <b>Iterasimus<b/>")
        yield from SplitMonstersLevelsFactory().get_intro_level()
        yield InfoLevel("You had no chance", "./images/apoke.16_00040.png", "you_had_no_chance", "You had no chance to beat <b>Iterasimus</b>") 
        yield InfoLevel("Oh... no", "./images/dream.jpg", "o_no", "Oh... no, you lost the battle. But the fight for the future is ongoing. We see the potential in you to became the greatest creator of all times, we will give you instructions how to prepare youself for the next night ...")

    def study_line_levels(self):
        yield InfoLevel("What a strange night", "./images/wake_up.jpg", "what_a_strange_night", "What a strange night! Let me prepare myself for the next battle")
        yield StudyLineLevel(StudyLineModel(0.0, -1.0, 0.0), StudyLineModel(0.0, -1.0, -0.5), [True, True, True, True, True, False], 1, 10)
        yield StudyLineLevel(StudyLineModel(1.0, 0.1, 0.0), StudyLineModel(1.0, 0.1, 0.5), [True, True, True, True, False, True], 2, 10)
        yield StudyLineLevel(StudyLineModel(1.0, 0.9, 0.0), StudyLineModel(1.0, 0.4, 0.0), [True, True, True, False, True, True], 3, 10)
        yield StudyLineLevel(StudyLineModel(1.0, -2.0, 0.0), StudyLineModel(1.0, -1.5, 0.0), [True, True, False, True, True, True], 4, 10)
        yield StudyLineLevel(StudyLineModel(-1.0, -2.0, 0.0), StudyLineModel(-1.5, -2.0, 0.0), [True, False, True, True, True, True], 5, 10)
        yield InfoLevel("You have no chance", "./images/apoke.16_00040.png", "you_have_no_chance", "You have no chance, I will win again ...")
        yield StudyLineLevel(StudyLineModel(1.0, -2.0, 0.0), StudyLineModel(1.5, -2.0, 0.0), [False, True, True, True, True, True], 6, 10)
        yield StudyLineLevel(StudyLineModel(0.5, 0.5, 0.0), StudyLineModel(0.7, 0.3, 0.0), [False, False, False, False, True, True], 7, 10)
        yield StudyLineLevel(StudyLineModel(0.5, 0.5, 0.0), StudyLineModel(0.7, 0.3, 0.5), [], 8, 10)
        yield StudyLineLevel(StudyLineModel(-0.5, -0.5, 0.0), StudyLineModel(0.7, 0.3, -0.3), [], 9, 10)
        yield StudyLineLevel(StudyLineModel(1.0, 0.3, 0.0), StudyLineModel(-1.0, 0.3, -0.7), [], 10, 10)

    def monster_levels(self):
        yield InfoLevel("Last night was crazy", "./images/sleep.jpg", "last_night", "Last night was crazy. I can not believe I spent half of the day in the training. What a silly move from my side! Time to go into the darkness and get some rest ...")
        yield InfoLevel("You are back", "./images/dream.jpg", "you_are_back", "You are back, we were waiting for you! Don't panic, you can win now. I know that you are not a real creator yet and you are scared, but I believe in you. Go and bring us a victory this time!")
        yield from SplitMonstersLevelsFactory().get_main_levels()
        yield InfoLevel("Congratulations", "./images/dream.jpg", "congratulations", "Congratulations! You earned your place among creators of the future. Now, you are ready to know what means to be a creator. By playing this game you actually studied machine learning. Join our secret group to continue your education and access to the next chapter of the game. Creators of the future are waiting for you https://www.facebook.com/groups/458107258671703")
