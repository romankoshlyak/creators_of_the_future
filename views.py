import copy
import numpy as np
from ipywidgets import Image, Layout, Button, VBox, HBox, Label, Box, GridBox, HTML, HTMLMath
from utils import WidgetsManager, Images, Sounds
from levels import LevelType, ErrorType
from level_factories import MainLevelsFactory
from models import LinearModel, StudyLineModel
from graphs import *
from actions import *

class LevelView(object):
    MINIMUM_LEARNING_RATE = 0.0001
    def __init__(self, level, main_view):
        self.widgets_manager = WidgetsManager()
        self.level = level
        self.main_view = main_view


    def set_up_learning_rate(self, level):
        self.learning_rate = self.level.learning_rate
        self.is_learning_rate_saved = False

    def get_model_data(self):
        data = torch.tensor([[point.x, point.y] for point in self.level.points], dtype=torch.float)
        return data

    def get_model_target(self):
        target = torch.tensor([point.target for point in self.level.points], dtype=torch.float)
        return target

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
        button.layout.width = '90%'
        button.on_click(action.do_action)
        if is_widget:
            self.widgets_manager.add_widget(button)
        return button

    def get_main_view(self, items):
        return GridBox(
            children=self.index_grid_items(items),
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='65% 35%',
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

    def __button_name_to_audio_file(self, button_name):
        sound_name = '_'.join(button_name.lower().split())
        return Sounds.get_file(sound_name)
        
    def get_all_spells(self):
        return []

    def get_magic_controls_items(self):
        items = self.get_all_spells()
        return items

    def get_magic_controls(self):
        items = self.get_magic_controls_items()
        return VBox(children=items)

    def get_weight_spells_items(self, model):
        param_index = 0
        items = []
        for param in model.parameters():
            tensor = param.data
            for i in range(len(tensor.view(-1))):
                param_name = self.PARAM_NAMES[param_index]
                param_index += 1
                button_name_add = f'{param_name} Adinimus'
                button_name_sub = f'{param_name} Subinimus'
                add_spell = self.__button_name_to_audio_file(button_name_add)
                sub_spell = self.__button_name_to_audio_file(button_name_sub)
                items.append(self.create_button(button_name_add, ChangeWeightAction(1.0, tensor, i, self).set_audio_file(add_spell)))
                items.append(self.create_button(button_name_sub, ChangeWeightAction(-1.0, tensor, i, self).set_audio_file(sub_spell)))
        return items

    def get_weight_spells(self, model):
        items = [Label('Iterasimus Spells:'), self.get_weight_spells_only(model)]
        return VBox(children=items)

    def get_weight_spells_only(self, model):
        items = self.get_weight_spells_items(model)
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_rows='repeat(4, max-content)',
                grid_template_columns='50% 50%')
       )

    def get_learning_rate_spells_items(self):
        items = [Label('Lernos Ratos Spells:')]
        items.append(self.create_button('Lernos Ratos Minisimus', SaveAndSetLearningRateAction(self, self.MINIMUM_LEARNING_RATE)))
        items.append(self.create_button('Lernos Ratos Restorisimus', RestoreLearningRateAction(self)))
        items.append(self.create_button('Lernos Ratos Incrisimus', ChangeLearningRateAction(self, 2.0)))
        items.append(self.create_button('Lernos Ratos Decrisimus', ChangeLearningRateAction(self, 0.5)))
        items = self.index_grid_items(items)
        return items

    def get_learning_rate_spells(self):
        items = self.get_learning_rate_spells_items()
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_columns='50% 50%',
                grid_template_areas='''
                "item0 item0"
                "item1 item2"
                "item3 item4"
                ''')
       )

    def get_learning_rate_controls_items(self):
        items = [Label('Learning rate:')]
        items.append(self.learning_rate_label)
        items.append(self.create_button('Increase', ChangeLearningRateAction(self, 10.0)))
        items.append(self.create_button('Descrease', ChangeLearningRateAction(self, 0.1)))
        items = self.index_grid_items(items)
        return items

    def get_learning_rate_controls(self):
        items = self.get_learning_rate_controls_items()
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_rows='repeat(4, max-content)',
                grid_template_columns='max-content auto auto',
                grid_template_areas='''
                "item0 item0 item0"
                "item1 item2 item3"
                ''')
       )

class StatusObject(object):
    def __init__(self, file_name, text_format, value, min_value = 0.0, max_value = 1.0):
        self.x_offset = 0.0
        self.img = Images.load_image(file_name)
        self.text_format = text_format
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.reverse = False
        self.extra_info = None

    def set_x_offset(self, x_offset):
        self.x_offset = x_offset

    def get_expected_length(self):
        return 1.0 + len(self.text_format.split("\\"))/10.0

    def set_reverse(self, reverse):
        self.reverse = reverse
        return self

    def set_max_value(self, max_value):
        self.max_value = max_value

    def set_extra_info(self, extra_info):
        self.exra_info = extra_info

class MonsterLevelView(LevelView):
    PARAM_NAMES = ['Witos Seros', 'Witos Unos', 'Bias']
    def __init__(self, level, main_view):
        super().__init__(level, main_view)
        self.set_up_learning_rate(level)
        self.learning_rate_object = StatusObject(Images.LEARNING_RATE_MONSTER, "Lenos Ratos\n {0:.4f}/{1:.2f}", 0.0).set_reverse(True)
        self.error = StatusObject(Images.ERROR_MONSTER, "Erorisimus\n {0:.6f}/{1:.2f}\n {2}", 0.0).set_reverse(True)
        self.accuracy = StatusObject(Images.ACCURACY_MONSTER, "Acurasimus\n {0:.2f}/{1:.2f}", 0.0)
        self.iteration = StatusObject(Images.ITERATION_MONSTER, "Iterasimus\n {0}/{1:.2f}", 0, 0, self.level.max_iterations)
        self.level_status = StatusObject(Images.LEVEL_MONSTER, "Level\n {0}/{1}", self.level.level_number, 0, self.level.number_of_levels)
        stats = [self.accuracy, self.iteration, self.level_status]
        if self.is_show_error():
            stats = [self.error] + stats
        if self.is_show_learning_rate():
            stats = [self.learning_rate_object] + stats
        self.bar_graph = BarGraph(stats)
        self.monster_min_size = 6.0
        self.monster_max_size = 10.0
        self.main_graph = MonsterGraph(self)
        self.update_learning_rate_label()
        # FIXME
        self.update_status(0.0, 0.0)

    def is_show_error(self):
        return False

    def is_show_learning_rate(self):
        return False

    def calc_monster_size(self):
        return (self.monster_max_size-self.monster_min_size)*(self.iteration.value)/float(self.level.max_iterations)+self.monster_min_size

    def update_learning_rate_label(self):
        self.learning_rate_object.value = self.learning_rate
        self.learning_rate_object.set_max_value(max(self.learning_rate_object.max_value, self.learning_rate))
        self.bar_graph.rerender()

    def update_status(self, error, accuracy):
        self.error.extra_info = f'{(error-self.error.value):+.6f}'
        self.error.value = error
        self.error.set_max_value(max(self.error.max_value, error))
        self.accuracy.value = accuracy
        self.bar_graph.rerender()

    def update_model(self):
        self.widgets_manager.disable_widgets()
        if self.iteration.value < self.level.max_iterations:
            if self.learning_rate > self.MINIMUM_LEARNING_RATE+1e-7:
                self.iteration.value += 1
            self.main_graph.rerender()
        if self.iteration.value < self.level.max_iterations:
            self.widgets_manager.enable_widgets()
        self.next_level_button.disabled = self.accuracy.value < 1.0-1e-7
        # First into fail level
        if self.level.hide_restart_button and self.iteration.value >= self.level.max_iterations:
            self.next_level_button.disabled = False

        self.bar_graph.rerender()

    def get_all_spells(self):
        return [self.get_learning_rate_spells(), self.get_weight_spells(self.level.model)]

    def render(self):
        show_restart_button = not self.level.hide_restart_button
        return self.get_main_view([self.bar_graph.get_graph(), self.get_level_controls(show_restart_button), self.main_graph.get_graph(), self.get_magic_controls()])

class MultiSplitMonsterLevelView(MonsterLevelView):
    def is_show_error(self):
        return True

class LearningRateMonstersLevelView(MultiSplitMonsterLevelView):
    def is_show_learning_rate(self):
        return True

class StudyLineLevelView(LevelView):
    PARAM_NAMES = ['Weight 0', 'Weight 1', 'Bias']
    def __init__(self, level, main_view):
        super(StudyLineLevelView, self).__init__(main_view)
        self.xgrid = (-3.0, 3.0)
        self.ygrid = (-3.0, 3.0)
        self.level = level
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

class StudyPlaneView(LevelView):
    PARAM_NAMES = ['Weight 0', 'Weight 1', 'Bias']
    def __init__(self, level, main_view):
        super().__init__(level, main_view)
        self.set_up_learning_rate(level)
        self.error = HTML('Error')
        self.error_value = 1.0
        self.main_graph = StudyPlaneGraph(self)

    def set_error(self, errors, colors):
        if self.level.error_type == ErrorType.SUM_LINEAR:
            error_name = "Sum linear error"
            part_format = "|{}|"
            all_format = "{}"
            error = sum(np.abs(errors))
        elif self.level.error_type == ErrorType.MEAN_LINEAR:
            error_name = "Mean linear error"
            part_format = "|{}|"
            all_format = "({})/{}"
            error = sum(np.abs(errors))/len(errors)
        elif self.level.error_type == ErrorType.SUM_SQUARED:
            error_name = "Sum squared error"
            part_format = "({})^2"
            all_format = "{}"
            error = sum(np.square(errors))
        elif self.level.error_type == ErrorType.MEAN_SQUARED:
            error_name = "Mean squared error"
            part_format = "({})^2"
            all_format = "({})/{}"
            error = sum(np.square(errors))/len(errors)

        parts = [part_format.format(f'<span style="color:{color}">{error:.2f}</span>') for error, color in zip(errors, colors)]
        self.error_value = error
        error = f'{error_name} ' + all_format.format('+'.join(parts), len(parts))  + f'={error:.2f} Limit={self.level.error_limit:.2f}'
        self.error.value = error

    def get_level_status(self):
        items = self.index_grid_items([HTML('<h1>Prepare yourself for the next night</h1>'), HTML(f'Level {self.level.level_number}/{self.level.number_of_levels}'), self.error])
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='20% 80%',
                grid_template_areas='''
                "item0 item0"
                "item1 item2"
                ''')
       )

    def get_controls_items(self, disabled_buttons = []):
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
        #self.disable_buttons(self.buttons, disabled_buttons)
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

    def update_model(self):
        self.widgets_manager.disable_widgets()
        self.main_graph.rerender()
        self.widgets_manager.enable_widgets()
        self.next_level_button.disabled = bool(self.error_value > self.level.error_limit)

    def render(self):
        return self.get_main_view([self.get_level_status(), self.get_level_controls(), self.main_graph.graph, self.get_controls()])

class InfoLevelView(LevelView):
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

class LearningRateLevelView(StudyPlaneView):
    GRAPHS = ['Error space', 'Problem space']
    def __init__(self, level, main_view):
        super().__init__(level, main_view)
        self.model = self.level.model
        self.data = self.get_model_data()
        self.target = self.get_model_target()
        self.loss = self.get_model_loss()
        self.learning_rate = level.learning_rate
        self.learning_rate_label = Label("Learning rate")
        self.graph_box = VBox(children=[])
        self.graph_selector = self.setup_graph_selector()

        self.update_error()
        self.update_learning_rate_label()

    def setup_graph_selector(self):
        is_error_space_3d = len(self.model.get_weights()) == 3
        error_space_graph = None
        if is_error_space_3d:
            error_space_graph = ErrorSpace3dGraph(self)
        else:
            error_space_graph = ErrorSpace2dGraph(self)
        graph_selector = widgets.Dropdown(options=self.GRAPHS, description='View:')

        action = SelectGraphAction(self, self.graph_box, graph_selector, [error_space_graph, StudyPlaneGraph(self)])
        action.do_action()
        graph_selector.observe(action.do_action, names='value')
        return graph_selector

    def update_learning_rate_label(self):
        self.learning_rate_label.value = f"{self.learning_rate:.5f}"

    def get_model_loss(self):
        if self.level.error_type == ErrorType.SUM_LINEAR:
            return nn.L1Loss(reduction='sum')
        raise f"Unsupported error_type: {self.level.error_type}"

    def set_error(self, output, target):
        if self.level.error_type == ErrorType.SUM_LINEAR:
            error_name = "Sum linear error"
            part_format = "|{}|"
            all_format = "{}"
            error = self.loss(output, target)
        else:
            raise f"Unsupported error_type: {self.level.error_type}"

        diff = output-self.target
        parts = [part_format.format(f'{d:.2f}') for d in diff.tolist()]
        self.error_value = error.item()
        error = f'{error_name} ' + all_format.format('+'.join(parts), len(parts))  + f'={error:.4f} Limit={self.level.error_limit:.4f}'
        self.error.value = error

    def update_error(self):
        output = self.model(self.data).view(-1)
        self.set_error(output, self.target)

    def update_model(self):
        self.widgets_manager.disable_widgets()
        self.current_graph.rerender()
        self.widgets_manager.enable_widgets()
        self.update_error()
        self.next_level_button.disabled = bool(self.error_value > self.level.error_limit+1e-5)

    def get_weight_controls_items(self):
        model = self.level.model
        param_index = 0
        items = [Label('Parameters:')]
        buttons = []
        for param in model.parameters():
            tensor = param.data
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
        items = self.index_grid_items(items)
        return items

    def get_weight_controls(self):
        items = self.get_weight_controls_items()
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

    def get_options_controls_items(self):
        items = [Label('Options:')]
        items.append(self.graph_selector)
        return items


    def get_options_controls(self):
        items = self.get_options_controls_items()
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_rows='repeat(4, max-content)',
                grid_template_columns='max-content auto auto',
                grid_template_areas='''
                "item0"
                "item1"
                ''')
       )

    def get_controls(self):
        return VBox(children=[self.get_options_controls(), self.get_learning_rate_controls(), self.get_weight_controls()])

    def get_level_status(self):
        items = self.index_grid_items([HTML('<h1>Prepare yourself for the next night</h1>'), HTML(f'Level {self.level.level_number}/{self.level.number_of_levels}'), self.error])
        return GridBox(
            children=items,
            layout=Layout(
                grid_template_rows='repeat(2, max-content)',
                grid_template_columns='20% 80%',
                grid_template_areas='''
                "item0 item0"
                "item1 item2"
                ''')
       )

    def render(self):
        return self.get_main_view([self.get_level_status(), self.get_level_controls(), self.graph_box, self.get_controls()])


class MainView(object):
    def __init__(self):
        self.levels = list(MainLevelsFactory().all_levels())
        self.main_box = VBox(children=[])
        self.load_current_level(0)

    def get_view_for_level(self, level):
        if level.level_type == LevelType.STUDY_LINE:
            return StudyLineLevelView(level, self)
        elif level.level_type == LevelType.INFO:
            return InfoLevelView(level, self)
        elif level.level_type == LevelType.SPLIT_MONSTERS:
            return MonsterLevelView(level, self)
        elif level.level_type == LevelType.STUDY_PLANE:
            return StudyPlaneView(level, self)
        elif level.level_type == LevelType.MULTI_SPLIT_MONSTERS:
            return MultiSplitMonsterLevelView(level, self)
        elif level.level_type == LevelType.LEARNING_RATE:
            return LearningRateLevelView(level, self)
        elif level.level_type == LevelType.LEARNING_RATE_MONSTERS:
            return LearningRateMonstersLevelView(level, self)

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