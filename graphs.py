import warnings
import sys
import math
import copy
import torch
import numpy as np
import ipywidgets as widgets
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

class BaseGraph(object):
    def __init__(self):
        self.graph_update = widgets.IntSlider(max=100000)
        self.__graph = None

    def get_graph(self):
        if self.__graph is None:
            self.__graph = widgets.interactive_output(self.render_with_catch, {'a' : self.graph_update})
        return self.__graph

    def render_with_catch(self, a):
        try:
            self.render(a)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def rerender(self, *args):
        self.graph_update.value = self.graph_update.value + 1

class BaseMainGraph(BaseGraph):
    def __init__(self, view):
        super().__init__()
        self.setup_base_main(view)

    def setup_color(self):
        target = [point.target for point in self.view.level.points]
        self.min_z = min(target)
        self.max_z = max(target)
        self.__setup_color()

    def setup_base_main(self, view):
        self.view = view
        self.model = view.level.model
        self.min_x, self.max_x = -3.0, 3.0
        self.min_y, self.max_y = -3.0, 3.0
        self.step = 0.1

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

    def __setup_color(self):
        level = self.view.level
        self.plot_target = sorted(list(set([point.target for point in level.points])))
        limit = max(-self.plot_target[0], self.plot_target[1])

        target = self.__get_range(limit, level.step_size)
        cols = self.__get_colors(target[1:-1])
        levels = [(x+y)/2.0 for x, y in zip(target, target[1:])]
        levels = [levels[0]-1000.0] + levels + [levels[-1]+1000.0]
        self.plot_cmap = colors.ListedColormap(cols)
        assert(len(levels)-1==self.plot_cmap.N)
        self.plot_norm = colors.BoundaryNorm(levels, self.plot_cmap.N)
        self.plot_levels = levels

    def np_points(self, grid):
        eps = 1e-7
        return np.arange(grid[0], grid[1]+eps, grid[2])

    def __make_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return x

    def get_color(self, x, norm=True):
        if norm:
            x = self.plot_norm(x)
        return self.plot_cmap(x)

    def model_prediction(self, X, Y):
        X = self.__make_tensor(X)
        Y = self.__make_tensor(Y)
        data = torch.cat((X.view(-1, 1), Y.view(-1, 1)), dim = 1)
        with torch.no_grad():
            output = self.model(data)
            return output.view_as(X).numpy()

    def get_model_target(self):
        target = torch.tensor([point.target for point in self.view.level.points], dtype=torch.float)
        return target

    def get_model_data_all(self):
        data = torch.tensor([point.vector for point in self.view.level.points], dtype=torch.float)
        return data

    def get_model_data(self):
        data = self.get_model_data_all()
        return data[:, 0], data[:, 1]

    def get_meshgrid(self):
        xgrid = (self.min_x, self.max_x, self.step)
        ygrid = (self.min_y, self.max_y, self.step)
        X, Y = np.meshgrid(self.np_points(xgrid), self.np_points(ygrid))
        return X, Y

    def set_borders(self, ax):
        ax.set_xlim([self.min_x, self.max_x])
        ax.set_ylim([self.min_y, self.max_y])
        ax.set_zlim([self.min_z, self.max_z])
        x_offset = (self.max_x-self.min_x)*0.02
        y_offset = (self.max_y-self.min_y)*0.02
        z_offset = (self.max_z-self.min_z)*0.02

        return self.min_x-x_offset, self.max_x+x_offset, self.min_y-y_offset, self.max_y+y_offset, self.min_z-z_offset, self.max_z+z_offset


class BarGraph(BaseGraph):
    def __init__(self, statuses):
        super().__init__()
        self.length = 10
        self.statuses = statuses
        self.set_statuses_offset()

    def set_statuses_offset(self):
        size_sum = sum([status.get_expected_length() for status in self.statuses])
        total = self.length
        offset = 0.0
        total_spaces = total-2*offset-size_sum
        each_space = total_spaces / (len(self.statuses)-1)
        x_offset = offset
        for status in self.statuses:
            status.set_x_offset(x_offset)
            x_offset += status.get_expected_length() + each_space

    def render(self, a):
        fig = plt.figure(figsize=(self.length, 1), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim([0, self.length])
        ax.set_ylim([0, 1])
        plt.axis('off')

        for status in self.statuses:
            ax.imshow(status.img, extent=[status.x_offset, status.x_offset+1, 0, 1], zorder=1)
            y_offset = (status.value-status.min_value)/(status.max_value-status.min_value)
            if status.reverse:
                y_offset = 1.0-y_offset
            rect = plt.Rectangle((status.x_offset, y_offset), 1, (1.0-y_offset), color='white', alpha=0.8, zorder=2)
            ax.add_patch(rect)
            args = [status.value, status.max_value]
            if status.extra_info is not None:
                args.append(status.extra_info)
            ax.text(status.x_offset+1, 0.5, status.text_format.format(*args))

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

class StudyPlaneGraph(BaseMainGraph):
    def __init__(self, view):
        super().__init__(view)
        self.options = StudyPlaneGraphOptions()
        self.setup_color()

    def set_labels(self, ax):
        first_dim_name = 'X'
        second_dim_name = 'Y'
        output_name = 'O1'
        title = "W0*{}+W1*{}+B".format(first_dim_name, second_dim_name)
        ax.set_title("{} -> {}".format(title, output_name))
        ax.set_xlabel(first_dim_name)
        ax.set_ylabel(second_dim_name)
        ax.set_zlabel(output_name)

    def render(self, a):
        x, y = self.get_model_data()
        z = self.model_prediction(x, y)
        X, Y = self.get_meshgrid()
        Z = self.model_prediction(X, Y)
        self.min_z = min(math.floor(np.min(Z)), self.min_z)
        self.max_z = max(math.ceil(np.max(Z)), self.max_z)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection='3d')

        self.set_labels(ax)
        min_x, max_x, min_y, max_y, min_z, max_z = self.set_borders(ax)

        if self.options.show_surface_projection:
            ax.contourf(X, Y, Z, offset=min_z, cmap=self.plot_cmap, norm=self.plot_norm, levels=self.plot_levels, alpha=0.5)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No contour levels were found within the data range.") 
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

class MonsterGraph(BaseMainGraph):
    def __init__(self, view):
        super().__init__(view)
        self.setup_color()
        self.hide_spell = view.level.hide_spell

    def draw_point(self, ax, monster, output_level, target_level):
        m_size = self.view.calc_monster_size()
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

    def draw_hide_spell(self, ax):
        ax.text(0.0, 0.0, "Iterasimum casted hide spell, you can not see battle field")

    def render(self, a):
        x, y = self.get_model_data()
        z = self.model_prediction(x, y)
        X, Y = self.get_meshgrid()
        Z = self.model_prediction(X, Y)

        fig = plt.figure(figsize=(8, 8), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        ax.set_xlim([self.min_x, self.max_x])
        ax.set_ylim([self.min_y, self.max_y])

        good = 0
        total = 0
        output_levels = self.plot_norm(z)
        level = self.view.level
        for index, point in enumerate(level.points):
            target_level = self.plot_norm(point.target)
            output_level = output_levels[index]
            good += 1 if target_level == output_level else 0
            total += 1
        accuracy = good/total
        if accuracy+1e-7 > 1.0:
            self.hide_spell = False

        if self.hide_spell:
            self.draw_hide_spell(ax)
        else:
            ax.contourf(X, Y, Z, cmap=self.plot_cmap, norm=self.plot_norm, levels=self.plot_levels, alpha=1.0)
            level = self.view.level
            for index, point in enumerate(level.points):
                target_level = self.plot_norm(point.target)
                output_level = output_levels[index]
                self.draw_point(ax, point, output_level, target_level)

        plt.show()
        error = sum([abs(z[i]-point.target) for i, point in enumerate(self.view.level.points)])
        self.view.update_status(error, good/total)

class ErrorSpaceBaseGraph(BaseMainGraph):
    def __init__(self, view):
        super().__init__(view)
        self.levels = [8.0, 4.0, 2.0]
        self.colors = ['blue', 'green', 'red']
        self.linestyles = [(0, (5, 5)), (0, (5, 5)), (0, (5, 5))]

    def get_model_data(self):
        model = copy.deepcopy(self.model)
        data = self.get_model_data_all()
        target = self.get_model_target()
        loss = nn.L1Loss(reduction='sum')
        return (model, data, target, loss)

    def func_array(self, model_data, X, Y, point, index1, index2):
        point = copy.deepcopy(point)
        res_shape = X.shape
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        V = np.empty_like(X)
        n = len(X)
        for i in range(n):
            point[index1] = X[i]
            point[index2] = Y[i]
            V[i] = self.func_fast(model_data, point)
        return V.reshape(res_shape)

    def func_fast(self, model_data, point):
        model, data, target, loss = model_data
        model.set_weights(point)
        with torch.no_grad():
            output = model(data).view_as(target)
            return loss(output, target).item()

class ErrorSpace2dGraph(ErrorSpaceBaseGraph):
    def set_labels(self, ax, view):
        ax.set_title("Error landscape")
        ax.set_xlabel(view.PARAM_NAMES[0])
        ax.set_ylabel(view.PARAM_NAMES[1])

    def render(self, a):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        self.set_labels(ax, self.view)
        X, Y = self.get_meshgrid()
        model = self.model
        x, y = model.get_weights()
        z = 0
        point_color = 'red'

        model_data = self.get_model_data()
        ax.scatter(x, y, color=point_color)
        V = self.func_array(model_data, X, Y, z)
        mi = np.min(V)
        ma = np.max(V)
        for level, color, linestyle in zip(self.levels, self.colors, self.linestyles):
            if mi < level and level < ma:
                cs = ax.contour(X, Y, V, levels=[level], colors=[color], linestyles=[linestyle])
                ax.clabel(cs)

        plt.show()

class ErrorSpace3dGraph(ErrorSpaceBaseGraph):
    def __init__(self, view):
        super().__init__(view)
        self.__setup()

    def __setup(self):
        self.min_z = -3.0
        self.max_z = 3.0
        mult = 3
        xgrid = (self.min_x, self.max_x, self.step*mult)
        ygrid = (self.min_y, self.max_y, self.step*mult)
        zgrid = (self.min_z, self.max_z, self.step*mult)
        X, Y = np.meshgrid(self.np_points(xgrid), self.np_points(ygrid))
        Z = self.np_points(zgrid)
        V = np.repeat(np.expand_dims(X.copy(), axis=0), len(Z), axis=0)
        model_data = self.get_model_data()
        for zi, z in enumerate(Z):
            V[zi] = self.func_array(model_data, X, Y, z)
        self.X = X
        self.Y = Y
        self.Z = Z
        self.V = V

    def set_labels(self, ax, view):
        ax.set_title("Error landscape")
        ax.set_xlabel(view.PARAM_NAMES[0])
        ax.set_ylabel(view.PARAM_NAMES[1])
        ax.set_zlabel(view.PARAM_NAMES[2])

    def indexer(self, a, indexes):
        return [a[i] for i in indexes]

    def draw_projection(self, ax, model_data, grids, zdirs, offsets, point, levels, colors, point_color, ind):
        M1, M2 = np.meshgrid(self.np_points(grids[ind[0]]), self.np_points(grids[ind[1]]))
        V = self.func_array(model_data, M1, M2, point[ind[2]], ind)
        mi = np.min(V)
        ma = np.max(V)
        for level, color in zip(levels, colors):
            if mi < level and level < ma:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
                    ax.contour(*self.indexer([M1, M2, V], ind), zdir=zdirs[ind[2]], offset=offsets[ind[2]], levels=[level], colors=[color])

        point_pr = list(point)
        point_pr[ind[2]] = offsets[ind[2]]
        line = list([[p,p_pr] for p, p_pr in zip(point, point_pr)])
        ax.scatter(*point_pr, color=point_color)
        ax.plot(*line, linestyle=(0, (5, 10)), c='grey')

    def render(self, a):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection='3d')
        self.set_labels(ax, self.view)
        min_x, max_x, min_y, max_y, min_z, max_z = self.set_borders(ax)

        for zi, z in enumerate(self.Z):
            mi = np.min(self.V[zi])
            ma = np.max(self.V[zi])
            for level, color, linestyle in zip(self.levels, self.colors, self.linestyles):
                if mi < level and level < ma:
                    ax.contour(self.X, self.Y, self.V[zi], offset=z, levels=[level], colors=[color], linestyles=[linestyle])

        model = self.model
        x, y, z = model.get_weights()

        point_color = 'red'
        ax.scatter(x, y, z, color=point_color)
        xgrid = (self.min_x, self.max_x, self.step)
        ygrid = (self.min_y, self.max_y, self.step)
        zgrid = (self.min_z, self.max_z, self.step)
        model_data = self.get_model_data()
        grids = (xgrid, ygrid, zgrid)
        zdirs = ('x', 'y', 'z')
        offsets = (min_x, max_y, min_z)
        point = (x, y, z)
        index_data = (grids, zdirs, offsets, point)
        args = (ax, model_data, *index_data, self.levels, self.colors, point_color)
        self.draw_projection(*args, np.array([0, 1, 2]))
        self.draw_projection(*args, np.array([0, 2, 1]))
        self.draw_projection(*args, np.array([2, 1, 0]))

        plt.show()

class Point2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def zero(cls):
        return cls(0.0, 0.0)

    @classmethod
    def from_vector(cls, vector):
        return cls(vector[0], vector[1])

    def mult(self, mult):
        return Point2D(self.x*mult, self.y*mult)

    def add(self, point):
        return Point2D(self.x+point.x, self.y+point.y)

    def neg(self):
        return Point2D(-self.x, -self.y)

    def right(self):
        return Point2D(self.y, self.x)

    def left(self):
        return Point2D(self.y, -self.x)

    def __iter__(self):
       return iter(self.to_list())

    def to_list(self):
        return [self.x, self.y]


class HigherDimensionsGraph(ErrorSpaceBaseGraph):
    def __init__(self, view):
        self.__setup_options()
        super().__init__(view)

    def plot_line(self, ax, p1, p2, color='black', linestyle = 'solid'):
        ax.plot((p1.x, p2.x), (p1.y, p2.y), color=color, linestyle=linestyle)

    def draw_projected_value(self, ax, axis, grid, transform,  point, model_data, index1, index2):
        offset = self.calc_offset(axis, index1, index2)
        ax1 = axis[index1]
        ax2 = axis[index2]
        transform = transform+transforms.Affine2D.from_values(ax1.x, ax1.y, ax2.x, ax2.y, 0.0, 0.0)+transforms.Affine2D().translate(offset.x, offset.y)+ax.transData
        M1, M2 = np.meshgrid(self.np_points(grid), self.np_points(grid))
        V = self.func_array(model_data, M1, M2, point, index1, index2)
        mi = np.min(V)
        ma = np.max(V)
        for level, color in zip(self.levels, self.colors):
            if mi < level and level < ma:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
                    ax.contour(M1, M2, V, levels=[level], colors=[color], transform=transform)

    def draw_projected_values(self, ax, axis, grid, transform, point):
        model_data = self.get_model_data()
        for index2 in range(len(axis)):
            for index1 in range(index2):
                self.draw_projected_value(ax, axis, grid, transform, point, model_data, index1, index2)

    def draw_axi_name(self, ax, axi, axi_name, grid, offset, direction=1):
        next_offset = offset.add(axi)
        offset_behind = axi.left().mult(direction*0.01)
        xylabel = offset.add(next_offset).mult(0.5).add(offset_behind)
        xylabel_left = offset.add(offset_behind)
        xylabel_right = next_offset.add(offset_behind)
        p1 = Point2D.from_vector(ax.transData.transform_point(list(offset)))
        p2 = Point2D.from_vector(ax.transData.transform_point(list(next_offset)))
        diff = p2.add(p1.neg())
        rotn = np.degrees(np.arctan2(*diff.right()))
        va_value = 'top' if direction == 1 else 'bottom'
        ax.annotate(axi_name, xy=xylabel, ha='center', va=va_value, rotation=rotn, rotation_mode='anchor')
        ax.annotate(f'{grid[0]:.2f}', xy=xylabel_left, ha='left', va=va_value, rotation=rotn, rotation_mode='anchor')
        ax.annotate(f'{grid[1]:.2f}', xy=xylabel_right, ha='right', va=va_value, rotation=rotn, rotation_mode='anchor')
        offset = next_offset
        return offset

    def draw_axis_names(self, ax, axis, axis_names, grid=(-1.0, 1.0)):
        axis_zip = list(zip(axis, axis_names))
        args = [(axis_zip, 1), (reversed(axis_zip), -1)]
        for arg_axis_zip, arg_direction in args:
            offset = Point2D.zero()
            for axi, axi_name in arg_axis_zip:
                offset = self.draw_axi_name(ax, axi, axi_name, grid, offset, direction=arg_direction)

    def draw_space(self, ax, axis, main_offset=Point2D(0.0, 0.0), color='black', linestyle='solid'):
        if len(axis) == 1:
            main_point = max(main_offset.to_list())
            self.plot_line(ax, main_offset, main_offset.add(axis[0]), color, linestyle)

        for index2 in range(len(axis)):
            for index1 in range(index2):
                offset = main_offset.add(self.calc_offset(axis, index1, index2))
                a1 = axis[index1]
                a2 = axis[index2]
                a_sum = a1.add(a2)
                a1 = offset.add(a1)
                a2 = offset.add(a2)
                a_sum = offset.add(a_sum)
                if index1 == 0:
                    self.plot_line(ax, offset, a1, color, linestyle)
                    self.plot_line(ax, a1, a_sum, color, linestyle)
                self.plot_line(ax, offset, a2, color, linestyle)
                self.plot_line(ax, a2, a_sum, color, linestyle)

    def calc_offset(self, axis, index1, index2):
        offset = Point2D(0.0, 0.0)
        for index in range(index1+1, index2):
            offset = offset.add(axis[index])
        return offset

    def calc_point2d(self, axis, point):
        result = Point2D(0.0, 0.0)

        for axi, p in zip(axis, point):
            result = result.add(axi.mult(p))
        return result

    def get_projection(self, axis, index1, index2, point):
        offset = self.calc_offset(axis, index1, index2)
        indexes = [index1, index2]
        projection_point = self.calc_point2d([axis[i] for i in indexes], [point[i] for i in indexes])
        projection_point = projection_point.add(offset)
        return projection_point

    def reindex(self, index, selected_index):
        if index >= selected_index:
            index -= 1
        return index

    def draw_point(self, ax, axis, selected_index, point, color):
        main_point = self.calc_point2d(axis, point)
        if self.show_subspace.value:
            subspace_axis = self.remove(axis, selected_index)
            subspace_point = self.remove(point, selected_index)
            subspace_offset = self.get_axis_offset(axis, point, selected_index)

        for index2 in range(len(axis)):
            for index1 in range(index2):
                offset = self.calc_offset(axis, index1, index2)
                projection_point = self.get_projection(axis, index1, index2, point)
                subspace_projection_point = None
                if self.show_subspace.value and index1 != selected_index and index2 != selected_index:
                    subspace_index1 = self.reindex(index1, selected_index)
                    subspace_index2 = self.reindex(index2, selected_index)
                    subspace_projection_point = subspace_offset.add(self.get_projection(subspace_axis, subspace_index1, subspace_index2, subspace_point))

                from_point = main_point
                if self.show_projection_line.value:
                    if subspace_projection_point is not None:
                        self.plot_line(ax, from_point, subspace_projection_point, linestyle=(0, (2, 4)), color='#8888ff')
                        ax.scatter(*subspace_projection_point, color='#8888ff')
                        from_point = subspace_projection_point
                    self.plot_line(ax, from_point, projection_point, linestyle=(0, (2, 4)), color='grey')
                ax.scatter(*projection_point, color='grey')

        ax.scatter(*main_point, color=color)

    def get_axis_names(self, model):
        axis_names = []
        for param_name, param in reversed(list(model.named_parameters())):
            tensor = param.data.view(-1)
            tensor_lengh = len(tensor)
            for i in range(tensor_lengh):
                axis_name_label = self.view.get_param_name(param_name, i)
                axis_names.append(axis_name_label)
        return axis_names

    def get_axis(self, n, next_axis_percentage):
        axis = []
        for i in range(n):
            angle = -math.pi/2.0+math.pi*((1.0+i)/(n)*(1.0-next_axis_percentage)+(1.0+i)/(n+1)*next_axis_percentage)
            axis.append(Point2D(math.cos(angle), math.sin(angle)))
        return axis

    def remove(self, l, i):
        ll = list(l)
        ll.pop(i)
        return ll

    def get_axis_offset(self, axis, point, index):
        return axis[index].mult(point[index])

    def calc_box(self, axis):
        min_x, max_x, min_y, max_y = 0.0, 0.0, 0.0, 0.0
        current_point = Point2D(0.0, 0.0)

        for direction in [1, -1]:
            for axi in axis:
                current_point = current_point.add(axi.mult(direction))
                min_x = min(min_x, current_point.x)
                max_x = max(max_x, current_point.x)
                min_y = min(min_y, current_point.y)
                max_y = max(max_y, current_point.y)
        max_d = max(max_x-min_x, max_y-min_y)
        text_width = max_d*0.01
        min_x -= text_width
        max_x += text_width
        min_y -= text_width
        max_y += text_width
        max_d = max(max_x-min_x, max_y-min_y)
        max_x = min_x+max_d
        max_y = min_y+max_d
        return min_x, max_x, min_y, max_y

    def setup_limits(self, ax, axis):
        min_x, max_x, min_y, max_y = self.calc_box(axis)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        plt.axis('off')

    def get_tranform(self, n):
        n = (n*(n-1))/2
        grid_min = -1.0
        grid_max = 1.0
        grid = (grid_min, grid_max, (grid_max-grid_min)/int(50.0/math.sqrt(n)))
        translate = -grid[0]
        scale = 1.0/(grid[1]-grid[0])
        translate_and_scale = transforms.Affine2D().translate(translate, translate).scale(scale, scale)
        transform1d = lambda x: (x+translate)*scale
        return grid, translate_and_scale, transform1d

    def render(self, a):
        parameters = parameters_to_vector(reversed(list(self.view.level.model.parameters())))
        axis = self.get_axis(len(parameters), self.next_axis_percentage.value)
        axis_names = self.get_axis_names(self.view.level.model)
        grid, transform, transform1d = self.get_tranform(len(parameters))
        point = parameters.tolist()
        normalized_point = list(map(transform1d, point))
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot()
        self.setup_limits(ax, axis)

        selected_index = self.view.selected_index
        self.draw_space(ax, axis)
        if self.show_projected_value.value:
            self.draw_projected_values(ax, axis, grid, transform, point)
        self.draw_axis_names(ax, axis, axis_names)
        self.draw_point(ax, axis, selected_index, normalized_point, 'blue')
        if self.show_subspace.value:
            subspace_axis = self.remove(axis, selected_index)
            subspace_offset = self.get_axis_offset(axis, normalized_point, selected_index)
            self.draw_space(ax, subspace_axis, main_offset=subspace_offset, color='blue', linestyle=(0, (5, 10)))
        plt.show()

    def __setup_options(self):
        self.show_subspace = widgets.Checkbox(value=True, description='Show subspace')
        self.show_projection_line = widgets.Checkbox(value=True, description='Show projection line')
        self.show_projected_value = widgets.Checkbox(value=True, description='Show projected error')
        self.next_axis_percentage = widgets.FloatSlider(value=0.0, min=0, max=1.0, step=0.001,
            description='Next axis:', continuous_update=True, readout=True, readout_format='.3f')
        self.options = [self.show_subspace, self.show_projection_line, self.show_projected_value, self.next_axis_percentage]
        for option in self.options:
            option.observe(self.rerender, 'value')

    def get_options(self):
        return self.options


class DevGraph(BaseMainGraph):
    def __init__(self, view):
        super().__init__(view)

    def render(self, a):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        ax.scatter(0, 0)
        plt.show()
