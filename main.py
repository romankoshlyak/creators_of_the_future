from actions import RestartLevelAction, NextLevelAction
from views import MainView

class Application(object):
    def render(self):
        view = MainView()
        return view.render()