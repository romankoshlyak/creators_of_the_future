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

class SplitMonstersLevelsFactory(BaseLevelFactory):
    def get_monsters(self, target_model, points, step_size):
        outputs = self.get_outputs(target_model, points)
        self.check_outputs(outputs, step_size)
        return [MonsterInfo(x, y, z, Images.DINO_MONSTER) for (x, y), z in zip(points, outputs)]

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
