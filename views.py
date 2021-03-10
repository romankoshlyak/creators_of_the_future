import copy
import numpy as np
from ipywidgets import Layout, Button, VBox, HBox, Label, Box, GridBox
from utils import WidgetsManager
from levels import InfoLevel, StudyLineLevel
from models import StudyLineModel
from graphs import StudyLineGraph
from levels import LevelType
from actions import ChangeWeightAction, NextLevelAction, RestartLevelAction

class LevelView(object):
    def __init__(self, main_view):
        self.widgets_manager = WidgetsManager()
        self.main_view = main_view

    def do_next_level(self):
        self.main_view.do_next_level()

    def index_grid_items(self, items):
        for index in range(len(items)):
            items[index].layout.grid_area = "item{}".format(index)
        return items

    def create_button(self, name, action):
        button = Button(description=name)
        button.on_click(action.do_action)
        self.widgets_manager.add_widget(button)
        return button

    def get_level_controls(self, show_restart_button = True, disable_next_button = True):
        items = []
        if show_restart_button:
            items.append(self.create_button('Restart level', RestartLevelAction(self)))
        else:
            items.append(Label(""))
        self.next_level_button = self.create_button('Next level', NextLevelAction(self))
        if disable_next_button:
            self.next_level_button.disabled = True
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

class MonsterLevelView(LevelView):
    PARAM_NAMES = ['Witos Seros', 'Witos Unos', 'Bias']
    def __init__(self, level):
        super(MonsterLevelView, self).__init__()
        self.level = level
        self.accuracy = StatusObject(DARK_F, 2.0, "Acurasimus\n {0:.2f}/{1}", 0.0)
        self.iteration = StatusObject(LIGHT_F, 5.0, "Iterasimus\n {0}/{1}", 0, 0, 10)
        self.bar_graph = BarGraph([self.accuracy, self.iteration])
        self.main_graph = Graph(LinearModel(0.5, 0.5, 0.0), self)

    def update_status(self, accuracy):
        self.accuracy.value = accuracy
        self.iteration.value += 1
        self.bar_graph.rerender()

    def update_model(self):
        self.widgets_manager.disable_widgets()
        self.main_graph.rerender()
        self.widgets_manager.enable_widgets()
        
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
                add_spell = button_name_to_audio_file(button_name_add)
                sub_spell = button_name_to_audio_file(button_name_sub)
                buttons.append(self.create_button(button_name_add, ChangeWeightAction(add_spell, 1.0, tensor, i, self).do_action))
                buttons.append(self.create_button(button_name_sub, ChangeWeightAction(sub_spell, -1.0, tensor, i, self).do_action))
                
        all_parameter_controls.append(widgets.GridBox(children=buttons, layout=Layout(grid_template_columns='auto auto')))
        return VBox(children=all_parameter_controls)

    def render(self):
        bar = self.bar_graph.graph
        main = self.main_graph.graph
        controls = self.get_controls(self.main_graph)
        return GridBox(
            children=self.index_grid_items([bar, main, controls]),
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='70% 30%',
                grid_template_areas='''
                "item0 item0"
                "item1 item2"
                "item1 ."
                ''')
       )

class StudyLineLevelView(LevelView):
    PARAM_NAMES = ['Witos Seros', 'Witos Unos', 'Bias']
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
        #if self.finished:
        #    self.game_status.description = "You prepared for next dream".format(self.current_level_index+1, len(self.levels))
        #    self.game_status.button_style = 'success'
        #else:
        self.game_status.description = "Level "#{}/{}".format(self.current_level_index+1, len(self.levels))
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

        return self.game_status_box

    def update_model(self):
        if self.is_same_line(self.level.model, self.level.target_model):
            self.next_level_button.disabled = False
        self.widgets_manager.disable_widgets()
        self.graph.rerender()
        if self.next_level_button.disabled:
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
        header = self.get_game_status()
        graph = self.graph.graph
        self.controls = self.get_controls()
        return GridBox(
            children=self.index_grid_items([header, self.get_level_controls(), graph, self.controls]),
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='70% 30%',
                grid_template_areas='''
                "item0 item1"
                "item2 item3"
                ''')
       )

class InfoLevelView(LevelView):
    def __init__(self, level, main_view):
        super(InfoLevelView, self).__init__(main_view)
        self.level = level


    def render(self):
        header = Label(self.level.text)
        info = Label(self.level.text)
        return GridBox(
            children=self.index_grid_items([header, self.get_level_controls(False, False), info]),
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='70% 30%',
                grid_template_areas='''
                "item0 item1"
                "item2 ."
                "item2 ."
                ''')
       )
        return Label(self.level.text)

class MainView(object):
    def __init__(self):
        self.levels = [
            InfoLevel("How are you?"),
            StudyLineLevel(StudyLineModel(0.0, -1.0, 0.0), StudyLineModel(0.0, -1.0, -0.5), [True, True, True, True, True, False]),
            InfoLevel("How are you?"),
            StudyLineLevel(StudyLineModel(1.0, 0.1, 0.0), StudyLineModel(1.0, 0.1, 0.5), [True, True, True, True, False, True]),
            StudyLineLevel(StudyLineModel(1.0, 0.9, 0.0), StudyLineModel(1.0, 0.4, 0.0), [True, True, True, False, True, True]),
            StudyLineLevel(StudyLineModel(1.0, -2.0, 0.0), StudyLineModel(1.0, -1.5, 0.0), [True, True, False, True, True, True]),
            StudyLineLevel(StudyLineModel(-1.0, -2.0, 0.0), StudyLineModel(-1.5, -2.0, 0.0), [True, False, True, True, True, True]),
            StudyLineLevel(StudyLineModel(1.0, -2.0, 0.0), StudyLineModel(1.5, -2.0, 0.0), [False, True, True, True, True, True]),
            StudyLineLevel(StudyLineModel(0.5, 0.5, 0.0), StudyLineModel(0.7, 0.3, 0.0), [False, False, False, False, True, True]),
            StudyLineLevel(StudyLineModel(0.5, 0.5, 0.0), StudyLineModel(0.7, 0.3, 0.5), []),
            StudyLineLevel(StudyLineModel(-0.5, -0.5, 0.0), StudyLineModel(0.7, 0.3, -0.3), []),
            StudyLineLevel(StudyLineModel(1.0, 0.3, 0.0), StudyLineModel(-1.0, 0.3, -0.7), []),
        ]
        self.main_box = VBox(children=[])
        self.load_current_level(0)

    def get_view_for_level(self, level):
        if level.level_type == LevelType.STUDY_LINE:
            return StudyLineLevelView(level, self)
        elif level.level_type == LevelType.INFO:
            return InfoLevelView(level, self)

    def do_next_level(self):
        self.load_current_level(self.current_level_index+1)

    def load_current_level(self, index):
        self.current_level_index = index
        level = copy.deepcopy(self.levels[self.current_level_index])
        self.main_box.children = [self.get_view_for_level(level).render()]

    def render(self):
        return self.main_box