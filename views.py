import copy
import numpy as np
from ipywidgets import Image, Layout, Button, VBox, HBox, Label, Box, GridBox, HTML
from utils import WidgetsManager, Images, Sounds
from levels import InfoLevel, StudyLineLevel, SplitMonstersLevelsFactory
from models import LinearModel, StudyLineModel
from graphs import StudyLineGraph, BarGraph, MonsterGraph
from levels import LevelType
from actions import ChangeWeightAction, NextLevelAction, RestartLevelAction

class LevelView(object):
    def __init__(self, main_view):
        self.widgets_manager = WidgetsManager()
        self.main_view = main_view

    def do_next_level(self):
        self.main_view.do_next_level()

    def do_restart_level(self):
        self.main_view.do_restart_level()

    def index_grid_items(self, items):
        for index in range(len(items)):
            items[index].layout.grid_area = "item{}".format(index)
        return items

    def create_button(self, name, action, is_widget = True):
        button = Button(description=name)
        button.on_click(action.do_action)
        if is_widget:
            self.widgets_manager.add_widget(button)
        return button

    def get_main_view(self, items):
        return GridBox(
            children=self.index_grid_items(items),
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='70% 30%',
                grid_template_areas='''
                "item0 item1"
                "item2 item3"
                ''')
        )

    def get_level_controls(self, show_restart_button = True, disable_next_button = True, hide_next_button = False):
        items = []
        if show_restart_button:
            items.append(self.create_button('Restart level', RestartLevelAction(self), False))
        else:
            items.append(Label(""))
        self.next_level_button = self.create_button('Next level', NextLevelAction(self), False)
        if disable_next_button:
            self.next_level_button.disabled = True
        if hide_next_button:
            items.append(Label(""))
        else:
            items.append(self.next_level_button)

        items = self.index_grid_items(items)
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='50% 50%',
                grid_template_areas='''
                "item0 item1"
                ''')
       )

class StatusObject(object):
    def __init__(self, file_name, x_offset, text_format, value, min_value = 0.0, max_value = 1.0):
        self.img = Images.load_image(file_name)
        self.x_offset = x_offset
        self.text_format = text_format
        self.value = value
        self.min_value = min_value
        self.max_value = max_value

class MonsterLevelView(LevelView):
    PARAM_NAMES = ['Witos Seros', 'Witos Unos', 'Bias']
    def __init__(self, level, main_view):
        super(MonsterLevelView, self).__init__(main_view)
        self.level = level
        self.accuracy = StatusObject(Images.ACCURACY, 2.0, "Acurasimus\n {0:.2f}/{1}", 0.0)
        self.iteration = StatusObject(Images.ITERATION, 5.0, "Iterasimus\n {0}/{1}", 0, 0, self.level.max_iterations)
        self.bar_graph = BarGraph([self.accuracy, self.iteration])
        self.monster_min_size = 3.0
        self.monster_max_size = 10.0
        self.main_graph = MonsterGraph(self.level.model, self)

    def calc_monster_size(self):
        return (self.monster_max_size-self.monster_min_size)*(self.iteration.value)/float(self.level.max_iterations)+self.monster_min_size

    def update_status(self, accuracy):
        self.accuracy.value = accuracy
        self.bar_graph.rerender()

    def update_model(self):
        self.widgets_manager.disable_widgets()
        if self.iteration.value < self.level.max_iterations:
            self.iteration.value += 1
        self.main_graph.rerender()
        if self.iteration.value < self.level.max_iterations:
            self.widgets_manager.enable_widgets()
        self.next_level_button.disabled = self.accuracy.value < 1.0-1e-7
        # First into fail level
        if self.level.hide_restart_button and self.iteration.value >= self.level.max_iterations:
            self.next_level_button.disabled = False

        self.bar_graph.rerender()

    def button_name_to_audio_file(self, button_name):
        sound_name = '_'.join(button_name.lower().split())
        return Sounds.get_file(sound_name)
        
    def get_controls(self, graph):
        model = graph.model
        all_parameter_controls = []
        all_parameter_controls.append(Label('Spells:'))
        param_index = 0
        buttons = []
        for param in model.named_parameters():
            tensor = param[1].data
            for i in range(len(tensor.view(-1))):
                param_name = self.PARAM_NAMES[param_index]
                param_index += 1
                button_name_add = f'{param_name} Adinimus'
                button_name_sub = f'{param_name} Subinimus'
                add_spell = self.button_name_to_audio_file(button_name_add)
                sub_spell = self.button_name_to_audio_file(button_name_sub)
                buttons.append(self.create_button(button_name_add, ChangeWeightAction(1.0, tensor, i, self).set_audio_file(add_spell)))
                buttons.append(self.create_button(button_name_sub, ChangeWeightAction(-1.0, tensor, i, self).set_audio_file(sub_spell)))
                
        all_parameter_controls.append(GridBox(children=buttons, layout=Layout(grid_template_columns='auto auto')))
        return VBox(children=all_parameter_controls)

    def render(self):
        show_restart_button = not self.level.hide_restart_button
        return self.get_main_view([self.bar_graph.graph, self.get_level_controls(show_restart_button), self.main_graph.graph, self.get_controls(self.main_graph)])

class StudyLineLevelView(LevelView):
    PARAM_NAMES = ['Weight 0', 'Weight 1', 'Bias']
    def __init__(self, level, main_view):
        super(StudyLineLevelView, self).__init__(main_view)
        self.xgrid = (-3.0, 3.0)
        self.ygrid = (-3.0, 3.0)
        self.level = level
        self.widgets_manager = WidgetsManager()
        self.finished = False
        self.graph = StudyLineGraph(self)

    def get_controls_items(self):
        model = self.level.model
        param_index = 0
        items = [Label('Weights')]
        buttons = []
        for param in model.named_parameters():
            tensor = param[1].data
            for i in range(len(tensor.view(-1))):
                param_name = self.PARAM_NAMES[param_index]
                param_index += 1
                items.append(Label(param_name))
                button = self.create_button('Add', ChangeWeightAction(1.0, tensor, i, self))
                buttons.append(button)
                items.append(button)
                button = self.create_button('Sub', ChangeWeightAction(-1.0, tensor, i, self))
                buttons.append(button)
                items.append(button)
                
        self.buttons = buttons
        self.disable_buttons(self.buttons, self.level.disabled_buttons)
        items = self.index_grid_items(items)
        return items

    def get_controls(self):
        items = self.get_controls_items()
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_rows='repeat(4, max-content)',
                grid_template_columns='max-content auto auto',
                grid_template_areas='''
                "item0 item0 item0"
                "item1 item2 item3"
                "item4 item5 item6"
                "item7 item8 item9"
                ''')
       )

    def set_game_status_success(self):
        self.game_status.button_style = 'success'

    def update_game_status(self):
        self.game_status.description = "Level #{}/{}".format(self.level.level_number, self.level.number_of_levels)
        self.game_status.button_style = 'success'
        self.game_status.button_style = 'info'

    def get_game_status(self):
        game_name = Button(description='Prepare yourself for next dream', disabled=True, layout=Layout(width='100%'))
        self.game_status = Button(description="", disabled=True, layout=Layout(width='100%'))
        self.update_game_status()
        items = self.index_grid_items([game_name, self.game_status])
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='50% 50%',
                grid_template_areas='''
                "item0 item1"
                "item2 item2"
                ''')
       )

    def update_model(self):
        self.next_level_button.disabled = not self.is_same_line(self.level.model, self.level.target_model)
        self.widgets_manager.disable_widgets()
        self.graph.rerender()
        self.widgets_manager.enable_widgets()

    def disable_buttons(self, buttons, disabled_buttons):
        for disabled, button in zip(disabled_buttons, buttons):
            button.disabled = disabled

    def index_grid_items(self, items):
        for index in range(len(items)):
            items[index].layout.grid_area = "item{}".format(index)
        return items

    def inside(self, x, grid):
        return grid[0]-1e-7 < x and x < grid[1]+1e-7
    
    def inside_square(self, x, y):
        return self.inside(x, self.xgrid) and self.inside(y, self.ygrid)

    def not_zero(self, x):
        return x if x != 0.0 else 1e-7

    def add_point(self, points, point):
        for old_point in points:
            if self.same_vector(old_point, point):
                return points
        points.append(point)
        return points

    def get_line(self, model):
        w0 = self.not_zero(model.get_w0().item())
        w1 = self.not_zero(model.get_w1().item())
        b = model.get_b().item()
        # x*w0+y*w1+b
        # y = (x*w0+b)/w1
        x0, x1 = self.xgrid
        y0 = -(x0*w0+b)/w1
        y1 = -(x1*w0+b)/w1
        points = []
        if self.inside_square(x0, y0):
            self.add_point(points, [x0, y0])
        if self.inside_square(x1, y1):
            self.add_point(points, [x1, y1])
        y0, y1 = self.ygrid
        x0 = -(y0*w1+b)/w0
        x1 = -(y1*w1+b)/w0
        if self.inside_square(x0, y0):
            self.add_point(points, [x0, y0])
        if self.inside_square(x1, y1):
            self.add_point(points, [x1, y1])
        if len(points) == 2:
            p0, p1 = points
            x0, y0 = p0
            x1, y1 = p1
        return [x0, x1, y0, y1]

    def get_normal_points(self, model):
        x0, x1, y0, y1 = self.get_line(model)
        xm = (x0+x1)/2.0
        ym = (y0+y1)/2.0
        xv = (y1-ym)/3.0
        yv = -(x1-xm)/3.0
        return [xm+xv, ym+yv, xm-xv, ym-yv]

    def same_vector(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        diff = v1-v2
        return diff.T.dot(diff) < 1e-3

    def get_normal_model(self, model):
        w0 = model.get_w0().item()
        w1 = model.get_w1().item()
        b = model.get_b().item()
        v = np.array([w0, w1, b])
        max_v = np.absolute(v).max()
        if abs(max_v) > 1e-7:
            v /= max_v
        return v
    
    def get_sign(self, model, x, y):
        w0 = model.get_w0().item()
        w1 = model.get_w1().item()
        b = model.get_b().item()
        if w0*x+w1*y+b > 0:
            return '+'
        else:
            return '_'

    def is_same_line(self, model1, model2):
        v1 = self.get_normal_model(model1)
        v2 = self.get_normal_model(model2)
        return self.same_vector(v1, v2)

    def render(self):
        return self.get_main_view([self.get_game_status(), self.get_level_controls(), self.graph.graph, self.get_controls()])

class InfoLevelView(LevelView):
    def __init__(self, level, main_view):
        super(InfoLevelView, self).__init__(main_view)
        self.level = level

    def render(self):
        header = Label(self.level.header)
        image = Image(
            value=self.level.image,
            format='jpg',
            width=768,
            height=576,
        )
        story = HTML(self.level.story)
        Sounds.play_audio(self.level.audio_file)

        return self.get_main_view([header, self.get_level_controls(False, False, self.level.hide_next_button), image, story])

class MainView(object):
    def __init__(self):
        self.levels = list(self.all_levels())
        self.main_box = VBox(children=[])
        self.load_current_level(0)

    def all_levels(self):
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
        yield InfoLevel("Congratulations", "./images/dream.jpg", "congratulations", "Congratulations! You earned your place among creators of the future. Now, you are ready to know what means to be a creator. By playing this game you actually studied machine learning. Join our secret group to continue your education and access to the next chapter of the game. Creators of the future are waiting for you https://www.facebook.com/groups/458107258671703").set_hide_next_button(True)

    def get_view_for_level(self, level):
        if level.level_type == LevelType.STUDY_LINE:
            return StudyLineLevelView(level, self)
        elif level.level_type == LevelType.INFO:
            return InfoLevelView(level, self)
        elif level.level_type == LevelType.SPLIT_MONSTERS:
            return MonsterLevelView(level, self)

    def do_next_level(self):
        self.load_current_level(self.current_level_index+1)

    def do_restart_level(self):
        self.load_current_level(self.current_level_index)

    def load_current_level(self, index):
        self.current_level_index = index
        level = copy.deepcopy(self.levels[self.current_level_index])
        self.main_box.children = [self.get_view_for_level(level).render()]

    def render(self):
        return self.main_box