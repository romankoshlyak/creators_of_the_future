import os
import math
import torch.nn as nn
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from ipywidgets import Layout, Button, VBox, HBox, Label, Box
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
PARAM_NAMES = ['Witos Seros', 'Witos Unos', 'Bias']
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

class Graph(object):
    def __init__(self, model, application, dim_count):
        self.m_size = 1
        self.model = model
        self.application = application
        self.dim_count = dim_count
        self.first_dim = self.dim_count-2
        self.second_dim = self.dim_count-1
        self.graph_update = widgets.IntSlider(max=100000)
        self.application.create_data(self)
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
        return self.application.data[:, [self.first_dim, self.second_dim]]

    def get_model_output(self):
        data = self.get_model_data()
        data = torch.from_numpy(data).float()
        with torch.no_grad():
            output = self.model(data)
        return output.view(-1).numpy()

    def get_coord_text(self, i):
        first_dim = self.first_dim
        second_dim = self.second_dim
        text = ''
        for ind in range(self.dim_count+1):
            c = self.application.data[i,ind]
            c_str = " {:.1f}".format(c)
            if ind == first_dim or ind == second_dim:
                c_str = '['+c_str+']'
            if ind == self.dim_count:
                text += '->'
            text += c_str

        return text

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
        fig = plt.figure(figsize=(12, 10))
        projection = '3d' if self.options.is_3d else None
        ax = fig.add_subplot(projection=projection)
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        if self.options.is_3d:
            ax.set_zlim([min_z, max_z])
        first_dim_name = self.application.dim_names[self.first_dim]
        second_dim_name = self.application.dim_names[self.second_dim]
        output_name = self.application.dim_names[self.dim_count]
        title = "W0*{}+W1*{}+B".format(first_dim_name, second_dim_name)
        if self.model.activation != LINEAR:
            title = "{}({})".format(self.model.activation, title)
        if self.options.is_axis_off:
            plt.axis('off')
        else:
            ax.set_title("{} -> {}".format(title, output_name))
            ax.set_xlabel(first_dim_name)
            ax.set_ylabel(second_dim_name)
        if self.options.is_3d:
            ax.set_zlabel(output_name)
        if self.options.is_surface_projection:
            z_offset=(max_z-min_z)*0.02
            #cs = ax.contourf(X, Y, Z, offset=min_z-z_offset, cmap='winter', alpha=0.8)
        if self.options.is_surface:
            if self.options.is_3d:
                ax.plot_surface(X, Y, Z, cmap='winter', alpha=0.1)
            elif self.options.is_2_colors:
                levels = [-100, 0, 100]
                cmap = colors.ListedColormap(['blue', 'yellow'])
                # the nuber of intervals must be equal to the number of listed colors
                assert(len(levels)-1==cmap.N)
                # the norm that we use to map values to colors, see the docs    
                norm = colors.BoundaryNorm(levels, cmap.N)
                ax.contourf(X, Y, Z, cmap=cmap, levels=levels, norm=norm)
            else:
                cs = ax.contourf(X, Y, Z, cmap='winter')
        if self.options.is_wireframe:
            ax.plot_wireframe(X, Y, Z, color='black', rcount=5, ccount=5)
        #cbar = fig.colorbar(cs)
        for i in range(len(data)):
            if self.options.is_point:
                if self.options.is_3d:
                    ax.scatter(x[i], y[i], z[i], s=50, marker=markers[i], c=colors[i])
                else:
                    #ax.scatter(x[i], y[i], marker=markers[i], c=colors[i])
                    img = MARK_TO_IMG[markers[i]]
                    point_target = data_target[i]
                    point_output = 1 if output[i] >= 0.0 else -1
                    target_color = 'yellow' if point_target == 1 else 'blue'
                    output_color = 'yellow' if point_output == 1 else 'blue'
                    self.draw_point(ax, img, x[i], y[i], target_color, output_color)

            if self.options.is_3d:
                if self.options.is_x_projection_point:
                    ax.scatter(min_x, y[i], z[i], marker=markers[i], c=colors[i])
                if self.options.is_x_projection_line:
                    ax.plot((min_x,x[i]), (y[i], y[i]), (z[i], z[i]), linestyle=(0, (5, 10)), c='grey')
                if self.options.is_y_projection_point:
                    ax.scatter(x[i], max_y, z[i], marker=markers[i], c=colors[i])
                if self.options.is_y_projection_line:
                    ax.plot((x[i],x[i]), (y[i], max_y), (z[i], z[i]), linestyle=(0, (5, 10)), c='grey')
                if self.options.is_z_projection_point:
                    ax.scatter(x[i], y[i], min_z-z_offset, marker=markers[i], c=colors[i])
                if self.options.is_z_projection_line:
                    ax.plot((x[i],x[i]), (y[i], y[i]), (min_z-z_offset, z[i]), linestyle=(0, (5, 10)), c='grey')

        #    ax.annotate(self.get_coord_text(i), xy=(x, y), ha="center", xytext=(0, 12), textcoords='offset points')
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
    def __init__(self, spell_audio_file, mult, tensor, index, application):
        self.mult = mult
        self.tensor = tensor
        self.index = index
        self.spell_audio_file = spell_audio_file
        self.application = application

    def do_action(self, *args):
        self.tensor.view(-1)[self.index] += self.mult * 0.1
        play_audio(self.spell_audio_file)
        self.application.rerender()

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

class Application(object):
    def __init__(self):
        self.widgets_manager = WidgetsManager()
        self.neurons_count = 3
        self.dim_names = ["X", "Y"] + ["N{}".format((ind+1)) for ind in range(self.neurons_count)]
        self.graphs = []
        self.graphs.append(Graph(LinearModel(0.5, 0.5, 0.0), self, len(self.graphs)+2))
        self.create_data()

    def get_dim_selector(self, graph):
        first_dim_selector = widgets.Dropdown(
            options=self.dim_names[:graph.dim_count],
            description='First dim:',
            index=graph.first_dim
        )
        first_dim_selector.observe(ChooseDimensionAction(self, graph, 0, first_dim_selector).do_action, names='value')
        self.widgets_manager.add_widget(first_dim_selector)
        second_dim_selector = widgets.Dropdown(
            options=self.dim_names[:graph.dim_count],
            description='Second dim:',
            index=graph.second_dim
        )
        second_dim_selector.observe(ChooseDimensionAction(self, graph, 1, second_dim_selector).do_action, names='value')
        self.widgets_manager.add_widget(second_dim_selector)
        return VBox(children=[first_dim_selector, second_dim_selector], layout=Layout(width='100%'))

    def create_button(self, name, on_click):
        button = widgets.Button(description=name)
        button.on_click(on_click)
        self.widgets_manager.add_widget(button)
        return button

        
    def get_controls(self, graph):
        model = graph.model
        all_parameter_controls = []
        all_parameter_controls.append(Label('Spells:'))
        param_index = 0
        buttons = []
        for param in model.named_parameters():
            tensor = param[1].data
            for i in range(len(tensor.view(-1))):
                param_name = PARAM_NAMES[param_index]
                param_index += 1
                button_name_add = f'{param_name} Adinimus'
                button_name_sub = f'{param_name} Subinimus'
                add_spell = button_name_to_audio_file(button_name_add)
                sub_spell = button_name_to_audio_file(button_name_sub)
                buttons.append(self.create_button(button_name_add, ChangeWeightAction(add_spell, 1.0, tensor, i, self).do_action))
                buttons.append(self.create_button(button_name_sub, ChangeWeightAction(sub_spell, -1.0, tensor, i, self).do_action))
                
        all_parameter_controls.append(widgets.GridBox(children=buttons, layout=Layout(grid_template_columns='auto auto')))

        return all_parameter_controls

    def create_data(self, current_graph=None):
        data = DATA
        graphs = self.graphs.copy()
        if current_graph is not None:
            graphs.append(current_graph)
        self.data = np.append(data, np.zeros([data.shape[0], len(graphs)]), 1)
        for ind, graph in enumerate(self.graphs):
            self.data[:, ind+2] = graph.get_model_output()

    def rerender(self):
        self.widgets_manager.disable_widgets()
        for ind, graph in enumerate(self.graphs):
            self.data[:, ind+2] = graph.get_model_output()
            graph.rerender()
        self.widgets_manager.enable_widgets()

    def render(self):
        boxes = []
        boxes.append(widgets.Label('Task: Split red and yellow objects. After last step, yellow objects should have value 1.0 and red objects should have value 0.0'))
        for graph in self.graphs:
            graph_widget = Box(children=[graph.graph], layout=Layout(width='70%'))
            controls = VBox(children=self.get_controls(graph), layout=Layout(width='30%'))
            box_layout = Layout(width='100%', height='100%')
            box = Box(children=[graph_widget, controls], layout=box_layout)
            boxes.append(box)
        return VBox(children=boxes)

app = Application()
app.render()