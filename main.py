import os
import math
import torch.nn as nn
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from ipywidgets import Layout, Button, VBox, HBox, Label, Box, GridBox
import ipywidgets as widgets
import torch
from google.colab import output
import base64

DREAM_DIR = './'
SOUND_DIR = os.path.join(DREAM_DIR, 'sounds')
IMAGES_DIR = os.path.join(DREAM_DIR, 'images')

DARK_F = './images/apoke.04_00067.png'
LIGHT_F = './images/apoke.16_00040.png'
MARK_TO_F = {
    'dark': DARK_F,
    'light': LIGHT_F
}
MARK_TO_IMG = { k : plt.imread(MARK_TO_F[k]) for k in MARK_TO_F}
DATA = np.array([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
data_target = [1, 1, -1, -1]
data_colors = ['red', 'blue', 'blue', 'red']
markers = ['dark', 'dark', 'light', 'light']
LINEAR = 'Linear'
SIGMOID = 'Sigmoid'
ACTIVATIONS = {
    LINEAR: None,
    SIGMOID: nn.Sigmoid(),
    'ReLU': nn.ReLU(),
    'Tanh': nn.Tanh()
}
class Level(object):
    def __init__(self, levels, colors):
        self.levels = levels
        self.colors = colors
        self.monsters = {}
        self.monster_positions = []

    def add_monster(self, name, file_name):
        self.monsters[name] = plt.imread(file_name)

    def add_monster_position(self, name, x, y, target_color):
        self.monster_positions.append((name, x, y, target_color))

def get_level():
    level = Level([-100, 0, 100], ['blue', 'yellow'])
    level.add_monster('blue', 'apoke.04_00067.png')
    level.add_monster('yellow', 'apoke.16_00040.png')
    level.add_monster_positions


class ButtonAction(object):
    def do_action(self, *args):
        pass

class ToggleButtonAction(ButtonAction):
    def __init__(self, d, key):
        self.d = d
        self.key = key

    def do_action(self, value):
        self.d[self.key] = value

class OpenAccordionAction(ButtonAction):
    def __init__(self, accordion):
        self.accordion = accordion

    def do_action(self, value):
        self.accordion.selected_index = 0 if value else None

class GraphOptions(object):
    def __init__(self):
        self.is_3d = False
        self.is_x_projection_point = False
        self.is_y_projection_point = False
        self.is_z_projection_point = False
        self.is_x_projection_line = False
        self.is_y_projection_line = False
        self.is_z_projection_line = False
        self.is_wireframe = False
        self.is_point = True
        self.is_surface = True
        self.is_surface_projection = False
        self.is_axis_off = True
        self.is_2_colors = True

    def create_toggle_button(self, application, value, name, on_click, on_click_rerender=True):
        button = widgets.ToggleButton(value, description=name)
        application.widgets_manager.add_widget(button)
        def on_value_change(change):
            on_click(button.value)
            if on_click_rerender:
                application.rerender()
        button.observe(on_value_change, names='value')
        return button

    def get_controls(self, application):
        d = self.__dict__
        buttons = []
        for k in sorted(d.keys()):
            buttons.append(self.create_toggle_button(application, d[k], k, ToggleButtonAction(d, k).do_action))
        accordion = widgets.Accordion(children=[VBox(children=buttons)], title=['Options'])
        accordion.selected_index = None
        open_accordion_buttion = self.create_toggle_button(application, False, 'Options', OpenAccordionAction(accordion).do_action, False)
        return widgets.VBox(children=[open_accordion_buttion, accordion])

class StatusObject(object):
    def __init__(self, file_name, x_offset, text_format, value, min_value = 0.0, max_value = 1.0):
        self.img = plt.imread(file_name)
        self.x_offset = x_offset
        self.text_format = text_format
        self.value = value
        self.min_value = min_value
        self.max_value = max_value

class BarGraph(object):
    def __init__(self, statuses):
        self.statuses = statuses
        self.graph_update = widgets.IntSlider(max=100000)
        self.graph = widgets.interactive_output(self.render, {'a' : self.graph_update})

    def rerender(self):
        self.graph_update.value = self.graph_update.value + 1

    def render(self, a):
        fig = plt.figure(figsize=(8, 1), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim([0, 8])
        ax.set_ylim([0, 1])
        plt.axis('off')
        for status in self.statuses:
            ax.imshow(status.img, extent=[status.x_offset, status.x_offset+1, 0, 1], zorder=1)
            y_offset = (status.value-status.min_value)/(status.max_value-status.min_value)
            rect = plt.Rectangle((status.x_offset, y_offset), 1, (1.0-y_offset), color='white', alpha=0.8, zorder=2)
            ax.add_patch(rect)
            ax.text(status.x_offset+1, 0.5, status.text_format.format(status.value, status.max_value))

        plt.show()

class Graph(object):
    def __init__(self, model, application,):
        self.m_size = 1
        self.model = model
        self.application = application
        self.graph_update = widgets.IntSlider(max=100000)
        self.options = GraphOptions()
        self.graph = widgets.interactive_output(self.render, {'a' : self.graph_update})

    def rerender(self):
        self.graph_update.value = self.graph_update.value + 1
        
    def np_points(self, grid):
        eps = 1e-7
        return np.arange(grid[0], grid[1]+eps, grid[2])        

    def model_prediction(self, X, Y):
        XT = torch.from_numpy(X).float()
        YT = torch.from_numpy(Y).float()
        data = torch.cat((XT.view(-1, 1), YT.view(-1, 1)), dim = 1)
        with torch.no_grad():
            output = self.model(data)
            return output.view_as(XT).numpy()

    def get_model_data(self):
        return DATA.copy()

    def get_model_output(self):
        data = self.get_model_data()
        data = torch.from_numpy(data).float()
        with torch.no_grad():
            output = self.model(data)
        return output.view(-1).numpy()

    def draw_point(self, ax, img, x, y, target_color, output_color):
        match_color = 'green' if target_color == output_color else 'red'
        target_circle = plt.Circle((x, y), 0.03*self.m_size, color=target_color, zorder=2)
        output_circle = plt.Circle((x, y), 0.04*self.m_size, color=output_color, zorder=2)
        correct_circle = plt.Circle((x, y), 0.05*self.m_size, color=match_color, zorder=2)
        ax.add_patch(correct_circle)
        ax.add_patch(output_circle)
        ax.add_patch(target_circle)
        ax.imshow(img, extent=[x-0.05*self.m_size, x+0.05*self.m_size, y-0.05*self.m_size, y+0.05*self.m_size], zorder=2)

    def render(self, a):
        self.m_size += 1
        data = self.get_model_data()
        x = data[:, 0]
        y = data[:, 1]
        output = self.model_prediction(x, y)
        min_x = math.floor(np.min(x)-1.0)
        max_x = math.ceil(np.max(x)+1.0)
        min_y = math.floor(np.min(y)-1.0)
        max_y = math.ceil(np.max(y)+1.0)
        xgrid = (min_x, max_x, 0.1)
        ygrid = (min_y, max_y, 0.1)
        X, Y = np.meshgrid(self.np_points(xgrid), self.np_points(ygrid))
        Z = self.model_prediction(X, Y)        
        min_z = math.floor(np.min(Z))
        max_z = math.ceil(np.max(Z))
        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])

        levels = [-100, 0, 100]
        cmap = colors.ListedColormap(['blue', 'yellow'])
        # the nuber of intervals must be equal to the number of listed colors
        assert(len(levels)-1==cmap.N)
        # the norm that we use to map values to colors, see the docs
        norm = colors.BoundaryNorm(levels, cmap.N)
        ax.contourf(X, Y, Z, cmap=cmap, levels=levels, norm=norm)
        good = 0
        total = 0
        for i in range(len(data)):
            img = MARK_TO_IMG[markers[i]]
            point_target = data_target[i]
            point_output = 1 if output[i] >= 0.0 else -1
            target_color = 'yellow' if point_target == 1 else 'blue'
            output_color = 'yellow' if point_output == 1 else 'blue'
            good += 1 if target_color == output_color else 0
            total += 1
            self.draw_point(ax, img, x[i], y[i], target_color, output_color)
        self.application.update_status(good/total)

        plt.show()

def play_audio(file_name):
    data = open(file_name, "rb").read()
    audio64 = base64.b64encode(data).decode('ascii')
    output.eval_js(f'new Audio("data:audio/wav;base64,{audio64}").play()')

def button_name_to_audio_file(button_name):
    file_name = '_'.join(button_name.lower().split())
    file_name = f'{file_name}.m4a'
    return os.path.join(SOUND_DIR, file_name)


class ChangeWeightAction(ButtonAction):
    def __init__(self, spell_audio_file, mult, tensor, index, view):
        self.mult = mult
        self.tensor = tensor
        self.index = index
        self.spell_audio_file = spell_audio_file
        self.view = view

    def do_action(self, *args):
        self.tensor.view(-1)[self.index] += self.mult * 0.1
        play_audio(self.spell_audio_file)
        self.view.update_model()

class ChooseDimensionAction(ButtonAction):
    def __init__(self, application, graph, dim_number, dim_selector):
        self.application = application
        self.graph = graph
        self.dim_number = dim_number
        self.dim_selector = dim_selector

    def do_action(self, *args):
        dim_index = self.dim_selector.index
        if self.dim_number == 0:
            self.graph.first_dim = dim_index
        else:
            self.graph.second_dim = dim_index
        self.application.rerender()

class LinearModel(nn.Module):
    def __init__(self, w1, w2, b1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(2, 1)
        self.linear.weight.data[0, 0] = w1
        self.linear.weight.data[0, 1] = w2
        self.linear.bias.data[0] = b1
        self.max_value = torch.tensor(-10.0)
        self.activation = LINEAR
 
    def set_activation(self, activation):
        self.activation = activation
        return self

    def forward(self, x):
        x = self.linear(x)
        activation = ACTIVATIONS[self.activation]
        if activation is not None:
            x = activation(x)
        return x

class WidgetsManager(object):
    def __init__(self):
        self.all_widgets = []
    
    def add_widget(self, widget):
        self.all_widgets.append(widget)

    def disable_widgets(self):
        self.saved_disabled = [widget.disabled for widget in self.all_widgets]
        for widget in self.all_widgets:
            widget.disabled = True

    def enable_widgets(self):
        for widget, disabled in zip(self.all_widgets, self.saved_disabled):
            widget.disabled = disabled

class LevelView(object):
    def __init__(self):
        self.widgets_manager = WidgetsManager()

    def index_grid_items(self, items):
        for index in range(len(items)):
            items[index].layout.grid_area = "item{}".format(index)
        return items

    def create_button(self, name, on_click):
        button = widgets.Button(description=name)
        button.on_click(on_click)
        self.widgets_manager.add_widget(button)
        return button

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

class Application(object):
    def __init__(self):
        pass

    def render(self):
        level = None
        view = MonsterLevelView(level)
        return view.render()

app = Application()
app.render()