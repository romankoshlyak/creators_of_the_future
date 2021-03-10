from actions import RestartLevelAction, NextLevelAction
from views import MainView

'''
LINEAR = 'Linear'
SIGMOID = 'Sigmoid'
ACTIVATIONS = {
    LINEAR: None,
    SIGMOID: nn.Sigmoid(),
    'ReLU': nn.ReLU(),
    'Tanh': nn.Tanh()
}

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
'''


class Application(object):
    def render(self):
        view = MainView()
        return view.render()