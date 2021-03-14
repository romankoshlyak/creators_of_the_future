from utils import Sounds

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

class ChangeWeightAction(ButtonAction):
    def __init__(self, mult, tensor, index, view):
        self.mult = mult
        self.tensor = tensor
        self.index = index
        self.view = view
        self.audio_file = None
    
    def set_audio_file(self, audio_file):
        self.audio_file = audio_file
        return self

    def do_action(self, *args):
        self.tensor.view(-1)[self.index] += self.mult * 0.1
        if self.audio_file is not None:
            Sounds.play_audio(self.audio_file)
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

class NextLevelAction(ButtonAction):
    def __init__(self, view):
        self.view = view

    def do_action(self, *args):
        self.view.do_next_level()

class RestartLevelAction(ButtonAction):
    def __init__(self, view):
        self.view = view

    def do_action(self, *args):
        self.view.do_restart_level()