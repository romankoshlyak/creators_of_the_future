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