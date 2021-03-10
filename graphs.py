import math
import torch
import numpy as np
import ipywidgets as widgets
from matplotlib import colors
import matplotlib.pyplot as plt

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

class MonsterGraph(object):
    def __init__(self, model, view):
        self.m_size = 1
        self.model = model
        self.view = view
        self.graph_update = widgets.IntSlider(max=100000)
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

    def get_model_output(self):
        data = self.get_model_data()
        data = torch.from_numpy(data).float()
        with torch.no_grad():
            output = self.model(data)
        return output.view(-1).numpy()

    def draw_point(self, ax, monster, output_level, colors):
        x = monster.x
        y = monster.y
        target_level = monster.target_level
        match_color = 'green' if monster.target_level == output_level else 'red'
        target_circle = plt.Circle((x, y), 0.03*self.m_size, color=colors[target_level], zorder=2)
        output_circle = plt.Circle((x, y), 0.04*self.m_size, color=colors[output_level], zorder=2)
        correct_circle = plt.Circle((x, y), 0.05*self.m_size, color=match_color, zorder=2)
        ax.add_patch(correct_circle)
        ax.add_patch(output_circle)
        ax.add_patch(target_circle)
        ax.imshow(monster.image, extent=[x-0.05*self.m_size, x+0.05*self.m_size, y-0.05*self.m_size, y+0.05*self.m_size], zorder=2)

    def render(self, a):
        self.m_size += 1
        x = np.array([monster.x for monster in self.view.level.monsters])
        y = np.array([monster.y for monster in self.view.level.monsters])
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

        level = self.view.level
        levels = level.levels
        cmap = colors.ListedColormap(level.colors)
        # the nuber of intervals must be equal to the number of listed colors
        assert(len(levels)-1==cmap.N)
        # the norm that we use to map values to colors, see the docs
        norm = colors.BoundaryNorm(levels, cmap.N)
        ax.contourf(X, Y, Z, cmap=cmap, levels=levels, norm=norm)
        good = 0
        total = 0
        output_levels = norm(output)
        for index, monster in enumerate(level.monsters):
            target_level = monster.target_level
            output_level = output_levels[index]
            good += 1 if target_level == output_level else 0
            total += 1
            self.draw_point(ax, monster, output_level, level.colors)
        self.view.update_status(good/total)

        plt.show()

class StudyLineGraph(object):
    def __init__(self, view):
        self.view = view
        self.graph_update = widgets.IntSlider(max=100000)
        self.graph = widgets.interactive_output(self.render, {'a' : self.graph_update})

    def rerender(self):
        self.graph_update.value = self.graph_update.value + 1

    def render(self, a):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot()
        if self.view.finished:
            ax.text(0.5, 0.5, 'You are ready for your dream now',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=20, color='red',
                transform=ax.transAxes)
            ax.set_axis_off()
        else:
            ax.set_xlim(self.view.xgrid)
            ax.set_ylim(self.view.ygrid)
            ax.set_title('X*W0+Y*W1+B=0')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            line = self.view.get_line(self.view.level.target_model)
            ax.plot(line[:2], line[2:], c='g')
            line = self.view.get_line(self.view.level.model)
            ax.plot(line[:2], line[2:], c='b')
            model = self.view.level.target_model
            x1, y1, x2, y2 = self.view.get_normal_points(model)
            ax.scatter(x1, y1, s=200, marker=self.view.get_sign(model, x1, y1), c='g', alpha=0.5)
            ax.scatter(x2, y2, s=200, marker=self.view.get_sign(model, x2, y2), c='g', alpha=0.5)
            model = self.view.level.model
            x1, y1, x2, y2 = self.view.get_normal_points(model)
            ax.scatter(x1, y1, s=200, marker=self.view.get_sign(model, x1, y1), c='b', alpha=0.5)
            ax.scatter(x2, y2, s=200, marker=self.view.get_sign(model, x2, y2), c='b', alpha=0.5)
        plt.show()