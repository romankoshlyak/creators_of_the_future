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
        self.tensor.view(-1)[self.index] += self.mult * self.view.learning_rate
        Sounds.play_audio(self.audio_file)
        self.view.update_model()

class ChangeLearningRateAction(ButtonAction):
    def __init__(self, view, mult):
        self.view = view
        self.mult = mult

    def do_action(self, *args):
        if not self.view.is_learning_rate_saved:
            self.view.learning_rate *= self.mult
            self.view.update_learning_rate_label()

class SaveAndSetLearningRateAction(ButtonAction):
    def __init__(self, view, value):
        self.view = view
        self.value = value

    def do_action(self, *args):
        if not self.view.is_learning_rate_saved:
            self.view.is_learning_rate_saved = True
            self.view.learning_rate_saved = self.view.learning_rate
            self.view.learning_rate = self.value
            self.view.update_learning_rate_label()

class RestoreLearningRateAction(ButtonAction):
    def __init__(self, view):
        self.view = view

    def do_action(self, *args):
        if self.view.is_learning_rate_saved:
            self.view.is_learning_rate_saved = False
            self.view.learning_rate = self.view.learning_rate_saved
            self.view.update_learning_rate_label()

class SelectGraphAction(ButtonAction):
    def __init__(self, view, graph_box, selector, graphs):
        self.view = view
        self.graph_box = graph_box
        self.selector = selector
        self.graphs = graphs

    def do_action(self, *args):
        self.view.current_graph = self.graphs[self.selector.index]
        self.view.current_graph.rerender()
        self.graph_box.children = [self.view.current_graph.get_graph()]

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