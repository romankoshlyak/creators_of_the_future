import os
import base64
from google.colab import output
import matplotlib.pyplot as plt

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
class Assets(object):
    DREAM_DIR = './'

class Images(Assets):
    IMAGES_DIR = os.path.join(Assets.DREAM_DIR, 'images')
    ERROR_MONSTER = os.path.join(IMAGES_DIR, 'apoke.15_00000.png')
    ACCURACY_MONSTER = os.path.join(IMAGES_DIR, 'apoke.04_00067.png')
    ITERATION_MONSTER = os.path.join(IMAGES_DIR, 'apoke.16_00040.png')
    LEVEL_MONSTER = os.path.join(IMAGES_DIR, 'apoke.22_00039.png')
    DINO_MONSTER = os.path.join(IMAGES_DIR, 'apoke.03_00046.png')
    SNOW_MONSTER = os.path.join(IMAGES_DIR, 'apoke.09_00038.png')

    @classmethod
    def load_image(cls, image_file):
        return plt.imread(image_file)

    @classmethod
    def load_image_file(cls, file_name):
        image_file = open(file_name, "rb")
        return image_file.read()

class Sounds(Assets):
    SOUND_DIR = os.path.join(Assets.DREAM_DIR, 'sounds')

    @classmethod
    def get_file(cls, sound_name):
        if sound_name is None:
            return None
        file_name = f'{sound_name}.m4a'
        return os.path.join(Sounds.SOUND_DIR, file_name)

    @classmethod
    def play_audio(cls, file_name):
        if file_name is not None:
            data = open(file_name, "rb").read()
            audio64 = base64.b64encode(data).decode('ascii')
            #output.eval_js(f'new Audio("data:audio/wav;base64,{audio64}").play()')
