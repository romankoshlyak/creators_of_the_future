import math
import torch
import numpy as np
import ipywidgets as widgets
from matplotlib import colors
import matplotlib.pyplot as plt

class BaseGraph(object):
    def __init__(self):
        self.graph_update = widgets.IntSlider(max=100000)
        self.graph = widgets.interactive_output(self.render, {'a' : self.graph_update})

    def rerender(self):
        self.graph_update.value = self.graph_update.value + 1

    def get_color(self, x, norm=True):
        if norm:
            x = self.plot_norm(x)
        return self.plot_cmap(x)

    def __get_colors(self, target):
        limit = max(abs(target[0]), abs(target[-1]))
        color_coef = 0.8/(2*limit)
        color_base = 0.1+limit*color_coef
        target = [color_coef * t + color_base for t in target]
        target = [0.0] + target + [1.0]
        color_cmap = plt.cm.get_cmap('viridis')
        return list(map(color_cmap, target))

    def __get_range(self, limit, step_size):
        target = []
        ind = -limit-step_size
        end_limit = limit+step_size+step_size*0.5
        while ind < end_limit:
            target.append(ind)
            ind += step_size
        return target

    def setup_color(self):
        level = self.view.level
        target = sorted(list(set([point.target for point in level.points])))
        limit = max(max(target), -min(target))

        target = self.__get_range(limit, level.step_size)
        cols = self.__get_colors(target[1:-1])
        levels = [(x+y)/2.0 for x, y in zip(target, target[1:])]
        levels = [levels[0]-1000.0] + levels + [levels[-1]+1000.0]
        self.plot_cmap = colors.ListedColormap(cols)
        assert(len(levels)-1==self.plot_cmap.N)
        self.plot_norm = colors.BoundaryNorm(levels, self.plot_cmap.N)
        self.plot_levels = levels

class BarGraph(BaseGraph):
    def __init__(self, statuses):
        self.statuses = statuses
        self.set_statuses_offset()
        super().__init__()

    def set_statuses_offset(self):
        size_sum = sum([status.get_expected_length() for status in self.statuses])
        total = 8.0
        offset = 0.5
        total_spaces = total-2*offset-size_sum
        each_space = total_spaces / (len(self.statuses)-1)
        x_offset = offset
        for status in self.statuses:
            status.set_x_offset(x_offset)
            x_offset += status.get_expected_length() + each_space

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

class MonsterGraph(BaseGraph):
    def __init__(self, model, view):
        self.model = model
        self.view = view
        super().__init__()
        
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

    def draw_point(self, ax, monster, output_level, target_level):
        m_size = self.view.calc_monster_size()*2
        x = monster.x
        y = monster.y
        match_color = 'green' if target_level == output_level else 'red'
        target_circle = plt.Circle((x, y), 0.03*m_size, color=self.get_color(target_level, False), zorder=2)
        output_circle = plt.Circle((x, y), 0.04*m_size, color=self.get_color(output_level, False), zorder=2)
        correct_circle = plt.Circle((x, y), 0.05*m_size, color=match_color, zorder=2)
        ax.add_patch(correct_circle)
        ax.add_patch(output_circle)
        ax.add_patch(target_circle)
        ax.imshow(monster.image, extent=[x-0.05*m_size, x+0.05*m_size, y-0.05*m_size, y+0.05*m_size], zorder=2)

    def render(self, a):
        self.setup_color()
        x = np.array([monster.x for monster in self.view.level.points])
        y = np.array([monster.y for monster in self.view.level.points])
        output = self.model_prediction(x, y)
        min_x, max_x = -3.0, 3.0
        min_y, max_y = -3.0, 3.0
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

        ax.contourf(X, Y, Z, cmap=self.plot_cmap, norm=self.plot_norm, levels=self.plot_levels, alpha=1.0)

        good = 0
        total = 0
        output_levels = self.plot_norm(output)
        level = self.view.level
        for index, point in enumerate(level.points):
            target_level = self.plot_norm(point.target)
            output_level = output_levels[index]
            good += 1 if target_level == output_level else 0
            total += 1
            self.draw_point(ax, point, output_level, target_level)
        plt.show()

        self.view.update_status(good/total)

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

class StudyPlaneGraphOptions(object):
    def __init__(self):
        self.show_target_line = True
        self.show_error = True
        self.show_error_line = True
        self.show_x_projection = False
        self.show_y_projection = True
        self.show_z_projection = True
        self.show_wireframe = True
        self.show_point = True
        self.show_surface = True
        self.show_surface_projection = True

class StudyPlaneGraph(BaseGraph):
    def __init__(self, view):
        self.view = view
        self.model = view.level.model
        self.options = StudyPlaneGraphOptions()
        target = [point.target for point in self.view.level.points]
        self.min_z = min(target)
        self.max_z = max(target)
        super().__init__()

    def np_points(self, grid):
        eps = 1e-7
        return np.arange(grid[0], grid[1]+eps, grid[2])

    def make_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return x

    def model_prediction(self, X, Y):
        X = self.make_tensor(X)
        Y = self.make_tensor(Y)
        data = torch.cat((X.view(-1, 1), Y.view(-1, 1)), dim = 1)
        with torch.no_grad():
            output = self.model(data)
            return output.view_as(X).numpy()

    def get_model_data(self):
        return torch.tensor([[point.x, point.y] for point in self.view.level.points], dtype=torch.float)

    def get_model_output(self):
        data = self.get_model_data()
        data = torch.from_numpy(data).float()
        with torch.no_grad():
            output = self.model(data)
        return output.view(-1).numpy()

    def render(self, a):
        self.setup_color()
        data = self.get_model_data()
        x = data[:, 0]
        y = data[:, 1]
        z = self.model_prediction(x, y)
        min_x, max_x = -3.0, 3.0
        min_y, max_y = -3.0, 3.0
        xgrid = (min_x, max_x, 0.1)
        ygrid = (min_y, max_y, 0.1)
        X, Y = np.meshgrid(self.np_points(xgrid), self.np_points(ygrid))
        Z = self.model_prediction(X, Y)        
        min_z = math.floor(np.min(Z))
        min_z = min(min_z, self.min_z)
        max_z = math.ceil(np.max(Z))
        max_z = max(max_z, self.max_z)
        self.min_z = min_z
        self.max_z = max_z
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])
        first_dim_name = 'X'
        second_dim_name = 'Y'
        output_name = 'O1'
        title = "W0*{}+W1*{}+B".format(first_dim_name, second_dim_name)
        ax.set_title("{} -> {}".format(title, output_name))
        ax.set_xlabel(first_dim_name)
        ax.set_ylabel(second_dim_name)
        ax.set_zlabel(output_name)
        x_offset = (max_x-min_x)*0.02
        y_offset = (max_y-min_y)*0.02
        z_offset = (max_z-min_z)*0.02
        min_x -= x_offset
        max_x += x_offset
        min_y -= y_offset
        max_y += y_offset
        min_z -= z_offset
        max_z += z_offset
        if self.options.show_surface_projection:
            ax.contourf(X, Y, Z, offset=min_z, cmap=self.plot_cmap, norm=self.plot_norm, levels=self.plot_levels, alpha=0.5)
            ax.contour(X, Y, Z, offset=min_z, cmap=self.plot_cmap, norm=self.plot_norm, levels=self.plot_target, alpha=1.0)
        if self.options.show_surface:
            ax.plot_surface(X, Y, Z, cmap=self.plot_cmap, norm=self.plot_norm, alpha=0.2)
        if self.options.show_wireframe:
            ax.plot_wireframe(X, Y, Z, color='black', rcount=5, ccount=5, linestyle=(0, (1, 2)))

        for i, point in enumerate(self.view.level.points):
            color = self.get_color(point.target)
            if self.options.show_point:
                ax.scatter(point.x, point.y, z[i], s=50, color=color)

            if self.options.show_target_line:
                ax.plot((min_x, min_x), (min_y, max_y), (point.target, point.target), color=color)
                ax.plot((min_x, max_x), (max_y, max_y), (point.target, point.target), color=color)

            if self.options.show_error_line:
                ax.plot((point.x, point.x), (max_y, max_y), (point.target, z[i]), linestyle=(0, (5, 5)), c='red')
                ax.scatter(point.x, max_y, point.target, c='red')

            if self.options.show_error:
                ax.text(point.x, max_y, point.target, f'{(z[i]-point.target):.2f}', 'x', c='red')

            if self.options.show_x_projection:
                ax.scatter(min_x, point.y, z[i], color=color)
                ax.plot((min_x,point.x), (point.y, point.y), (z[i], z[i]), linestyle=(0, (5, 10)), c='grey')

            if self.options.show_y_projection:
                ax.scatter(point.x, max_y, z[i], color=color)
                ax.plot((point.x,point.x), (point.y, max_y), (z[i], z[i]), linestyle=(0, (5, 10)), c='grey')

            if self.options.show_z_projection:
                ax.scatter(point.x, point.y, min_z, color=color)
                ax.plot((point.x,point.x), (point.y, point.y), (min_z, z[i]), linestyle=(0, (5, 10)), c='grey')

        plt.show()

        errors = [z[i]-point.target for i, point in enumerate(self.view.level.points)]
        cols = [colors.to_hex(self.get_color(point.target)) for point in self.view.level.points]
        self.view.set_error(errors, cols)
