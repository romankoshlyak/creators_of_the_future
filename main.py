import os
import math
import copy
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
from actions import RestartLevelAction, NextLevelAction
from views import MainView

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

def play_audio(file_name):
    data = open(file_name, "rb").read()
    audio64 = base64.b64encode(data).decode('ascii')
    output.eval_js(f'new Audio("data:audio/wav;base64,{audio64}").play()')

def button_name_to_audio_file(button_name):
    file_name = '_'.join(button_name.lower().split())
    file_name = f'{file_name}.m4a'
    return os.path.join(SOUND_DIR, file_name)

class Level(object):
    def __init__(self, model, target_model, disabled_buttons):
        self.model = model
        self.target_model = target_model
        self.disabled_buttons = disabled_buttons

class Application(object):
    def render(self):
        view = MainView()
        return view.render()