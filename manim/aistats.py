from manimlib.imports import *

import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm, colors, pyplot as plt

sys.path.append(os.path.join('..', 'src'))
from algraph import GraphThresholdActiveLearner as GrAPL

################################################################################
# constants
################################################################################

TITLE_SCALE = 1.4
SMALL_SCALE = 0.7
PATH_TO_IMAGES = 'images'
MARGIN_SCALE = 0.9
CMAP = colors.LinearSegmentedColormap.from_list('3b1b_bwr', [BLUE, WHITE, RED])
LINE_GRAPH_NUM_NODES = 30
TICK_WIDTH = 0.3
TICK_SCALE = SMALL_SCALE
COLOR_CYCLE = [BLUE, YELLOW, RED, GREEN]

################################################################################
# useful things
################################################################################

class Network(VGroup):

    def __init__(self, G, pos, node_vals=None, node_scale=0.2, graph_scale=None, **kwargs):
        
        self.G = G
        self.graph_scale = graph_scale
        self.pos = self.normalize_pos(pos)

        if node_vals is None:
            node_vals = {node: 0.5 for node in self.G.nodes}
        else:
            node_vals = {node: val for node, val in zip(self.G.nodes, node_vals)}
        self.node_vals = node_vals

        self.node_scale = node_scale

        nodes = []
        for node in self.G.nodes:
            circ = Circle(color=get_cmap_color(self.node_vals[node]), fill_opacity=1.0, fill_color=BLACK)
            circ.scale(self.node_scale)
            circ.move_to(get_3d_from_2d(self.pos[node]))
            nodes.append(circ)
        
        edges = []
        for u, v in self.G.edges:
            line = Line(get_3d_from_2d(self.pos[u]), get_3d_from_2d(self.pos[v]))
            edges.append(line)

        self.nodes = VGroup(*nodes, **kwargs)
        self.edges = VGroup(*edges, **kwargs)

        super().__init__(self.nodes, self.edges, **kwargs)

    def copy_with_cvals(self, node_vals=None):

        return self.__class__(self.G, self.pos, node_vals=node_vals, node_scale=self.node_scale, graph_scale=self.graph_scale)

    def normalize_pos(self, pos):
        
        if self.graph_scale is None:
            return pos.copy()
        else:
            minmax_scaler = UniformMinMaxScaler((-self.graph_scale, self.graph_scale)).fit(np.stack(list(pos.values()), 0))
            return {node: minmax_scaler.transform(coords[None, :]).ravel() for node, coords in pos.items()}

class UniformMinMaxScaler(TransformerMixin, BaseEstimator):

    def __init__(self, feature_range=(0,1), copy=True):
        self.minmax_scaler = MinMaxScaler(feature_range=feature_range, copy=copy)

    def fit(self, X, y=None):
        self.minmax_scaler.fit(X.reshape(-1, 1), y=y)
        return self
    
    def transform(self, X):
        shape = X.shape
        return np.reshape(self.minmax_scaler.transform(X.reshape(-1, 1)), shape)

class MPLPlot(VGroup):

    def __init__(self, x=None, y=None, figsize=(3, 2), color=WHITE, stroke_width=2.0, box=False, title=None, title_scale=1.0, ticks=False, ticks_scale=1.0, xlims=None, ylims=None, xscale='linear', yscale='linear', **kwargs):

        VGroup.__init__(self, **kwargs)

        fig, self.ax = plt.subplots(figsize=figsize)

        self.figsize = np.array(figsize)
        self.offset = np.zeros((2,))

        self.color = color
        self.stroke_width = stroke_width

        self.box = Rectangle(height=figsize[1], width=figsize[0], color=self.color, stroke_width=self.stroke_width)
        self.box.shift([figsize[0] / 2, figsize[1] / 2, 0])
        self.add(self.box)
        if not box:
            self.box.fade(1)

        self.set_xlims(xlims)
        self.set_ylims(ylims)

        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)

        self.curves = []
        if x is not None and y is not None:
            self.add_curve(x, y)
            self.add(*self.curves)

        if title is not None:
            self.title = TextMobject(title, color=self.color)
            self.title.scale(title_scale)
            self.title.next_to(self.box, UP)
            self.add(self.title)
        else:
            self.title = None

        if ticks is True:
            self.ticks = self.create_ticks(ticks_scale=ticks_scale)
            self.add(self.ticks)
        elif ticks is not None and ticks is not False:
            self.ticks = self.create_ticks(*ticks, ticks_scale=ticks_scale)
            self.add(self.ticks)
        else:
            self.ticks = None
    
    def axes_transform(self, t):
        return self.ax.transLimits.transform(self.ax.transScale.transform(t))

    def scene_transform(self, t):
        return self.figsize * np.array(t) + self.offset
    
    def transform(self, t):
        return self.scene_transform(self.axes_transform(t))
    
    def shift(self, *vectors):
        VMobject.shift(self, *vectors)
        if len(vectors) == 1:
            self.offset += vectors[0][:2]
    
    def create_ticks(self, ax_xticks=None, ax_yticks=None, ticks_scale=1.0):

        if ax_xticks is None:
            ax_xticks = [x for x in self.ax.get_xticks() if 0 <= self.axes_transform([x, 0])[0] <= 1]
        if ax_yticks is None:
            ax_yticks = [y for y in self.ax.get_yticks() if 0 <= self.axes_transform([0, y])[1] <= 1]

        xticks, yticks = VGroup(), VGroup()

        for x in ax_xticks:
            tick = Line(ORIGIN, UP * TICK_WIDTH * ticks_scale, color=self.color, stroke_width=self.stroke_width)
            tick.move_to(get_3d_from_2d(self.scene_transform((self.axes_transform([x, 0])[0], 0))))
            tick_label = TextMobject(get_scale_label(x, self.ax.get_xscale()))
            tick_label.scale(TICK_SCALE * ticks_scale)
            tick_label.next_to(tick, DOWN)
            xticks.add(VGroup(tick, tick_label))

        for y in ax_yticks:
            tick = Line(ORIGIN, RIGHT * TICK_WIDTH * ticks_scale, color=self.color, stroke_width=self.stroke_width)
            tick.move_to(get_3d_from_2d(self.scene_transform((0, self.axes_transform([0, y])[1]))))
            tick_label = TextMobject(get_scale_label(y, self.ax.get_yscale()))
            tick_label.scale(TICK_SCALE * ticks_scale)
            tick_label.next_to(tick, LEFT)
            yticks.add(VGroup(tick, tick_label))

        return VGroup(xticks, yticks)
    
    def set_xlims(self, xlims=None):
        self.set_lims(self.ax.set_xlim, xlims)

    def set_ylims(self, ylims=None):
        self.set_lims(self.ax.set_ylim, ylims)
                
    def set_lims(self, set_lim_function, lims=None):
        if lims is None:
            set_lim_function(auto=True)
        else:
            set_lim_function(*lims, auto=False)
    
    def new_curve(self, xdata, ydata):
        curve = VMobject(color=self.color, stroke_width=self.stroke_width)
        curve.xdata = xdata
        curve.ydata = ydata
        curve.set_points_smoothly([get_3d_from_2d(self.transform((t, z))) for t, z in zip(xdata, ydata)])
        return curve

    def add_curve(self, x, y, clip=True):
        self.ax.plot(x, y)
        curve = self.new_curve(x, y)
        if clip:
            points = curve.get_points()
            for j in [0, 1]:
                np.clip(points[:, j], self.offset[j], self.offset[j] + self.figsize[j], out=points[:, j])
            curve.set_points(points)
        self.curves.append(curve)
    
    def get_updated_curves(self, curves=None):

        if curves is None:
            curves = self.curves

        new_curves = VGroup()
        for curve in curves:
            new_curves.add(self.new_curve(curve.xdata, curve.ydata))
        return new_curves
    
    def animate_update_previous(self):
        return Transform(VGroup(*self.curves[:-1]), self.get_updated_curves(self.curves[:-1]))

    def animate_grow_last(self, animation=ShowCreation, **kwargs):
        self.add(self.curves[-1])
        return animation(self.curves[-1], **kwargs)
    
    def get_new_mpl_plot(self, color=None, **kwargs):
        
        # automatically cycle colors
        if color is None:
            if self.color == WHITE:
                color = COLOR_CYCLE[0]
            else:
                try:
                    i = COLOR_CYCLE.index(self.color)
                    color = COLOR_CYCLE[(i + 1) % len(COLOR_CYCLE)]
                except ValueError as e:
                    color = WHITE

        mpl = self.__class__(
            x=None, y=None, figsize=self.figsize, 
            color=color, stroke_width=self.stroke_width, 
            box=False, title=None, 
            ticks=False, 
            xlims=self.ax.get_xlim(), ylims=self.ax.get_ylim(), 
            xscale=self.ax.get_xscale(), yscale=self.ax.get_yscale(), 
            **kwargs
        )
        mpl.shift(get_3d_from_2d(self.offset))
        return mpl

def get_cmap_color(v):
    rgba = CMAP(v)
    return rgb_to_hex(rgba[:3])

def vector_to_0_1_range(v, scale=1):
    return np.clip(0.5 + v / np.linalg.norm(v) * np.sqrt(len(v)) * scale, 0, 1)

def get_3d_from_2d(v, z=0):
    return np.concatenate((v, [z]))

def get_scale_label(x, scale='linear'):
    
    if scale == 'log':
        z = np.log10(x)
        z_0 = np.rint(z)
        z_1 = np.around(z, 1)
        if np.isclose(z_0, z_1):
            return '$10^{%d}$' % z_0
        else:
            return '$10^{%.1f}$' % z_1
    else:
        return '%g' % x

################################################################################
# begin scenes
################################################################################

class MPLScene(Scene):
    """Scene for testing MPLPlot
    """

    def construct(self):
        
        ts = np.linspace(1, 5)
        mpl = MPLPlot(ts,
            np.sin(3 * ts),
            figsize=(12, 6), 
            box=True, 
            ticks=([1, 10**0.5, 10], 
            [-1, 0, 1]),
            xlims=(1, 10),
            xscale='log'
        )
        mpl.shift(6*LEFT + 3*DOWN)

        self.play(ShowCreation(mpl), run_time=3)

        for t in range(5, 11):
            more_ts = np.linspace(t, t+1)
            mpl = mpl.get_new_mpl_plot()
            mpl.add_curve(more_ts, np.sin(3 * more_ts))
            #self.play(mpl.animate_update_previous(), run_time=0.2)
            self.play(mpl.animate_grow_last(), run_time=0.2)


class TitleScene(Scene):

    def construct(self):

        title = TextMobject('Thresholding Graph Bandits with GrAPL')
        title.scale(TITLE_SCALE)
        subtitle = TextMobject('AISTATS 2020')
        title_vg = VGroup(title, subtitle)
        title_vg.arrange(direction=DOWN, center=True)

        self.play(Write(title_vg))
        self.wait()

        self.play(FadeOut(title_vg))
        self.wait()

class AuthorScene(Scene):

    def construct(self):

        authors = [
            ['Daniel LeJeune', 'daniel.jpg', 'Rice University'],
            ['Gautam Dasarathy', 'gautam.jpg', 'Arizona State University'],
            ['Richard G. Baraniuk', 'baraniuk.jpg', 'Rice University']
        ]

        photos = []
        author_affils = []

        for i, (author_name, photo_filename, university) in enumerate(authors):

            photo = ImageMobject(os.path.join(PATH_TO_IMAGES, photo_filename))
            photo.scale(MARGIN_SCALE)
            photo.move_to([-3, 2 - 2*i, 0], aligned_edge=RIGHT)
            photos.append(photo)

            name_mobject = TextMobject(author_name)
            univ_mobject = TextMobject(university)
            univ_mobject.scale(SMALL_SCALE)

            author_affil_vg = VGroup(name_mobject, univ_mobject)
            author_affil_vg.arrange(direction=DOWN, aligned_edge=LEFT)
            author_affil_vg.next_to(photo, direction=RIGHT)
            author_affils.append(author_affil_vg)

        self.play(
            LaggedStartMap(
                lambda x: LaggedStartMap(FadeIn, x, lag_ratio=0.1),
                Group(Group(*photos), VGroup(*author_affils)),
                lag_ratio=0.5
            )
        )
        self.wait()

class GrAPLLineGraphScene(Scene):

    @staticmethod
    def create_line_graph_with_values(vals, x_scale=6):

        n = len(vals)
        G = nx.path_graph(n)
        pos = {node: np.array([x, val]) for node, x, val in zip(G.nodes, np.linspace(-x_scale, x_scale, n), vals)}
        return Network(G, pos, node_vals=vals, node_scale=x_scale*2/3/LINE_GRAPH_NUM_NODES, graph_scale=None)

    def construct(self):
        
        G = nx.path_graph(LINE_GRAPH_NUM_NODES)
        L = nx.laplacian_matrix(G).toarray()
        np.random.seed(42)
        smooth_signal = np.linalg.solve(L + np.eye(LINE_GRAPH_NUM_NODES), np.random.randn(LINE_GRAPH_NUM_NODES))
        smooth_vals = vector_to_0_1_range(smooth_signal)

        true_net = self.create_line_graph_with_values(smooth_vals)
        true_net.center()

        true_dash_line = DashedLine([-6.1, 0, 0], [6.1, 0, 0])
        self.play(ShowCreation(true_dash_line))
        
        self.play(ShowCreation(true_net.nodes))
        self.bring_to_back(true_net.edges)
        self.play(FadeIn(true_net.edges))

        # move true net up and create working net below

        true_group = Group(true_dash_line, true_net.edges, true_net.nodes)
        self.play(true_group.move_to, UP * 1.5)

        working_net = self.create_line_graph_with_values(0.5 + np.zeros_like(smooth_vals))
        working_net.shift(DOWN * 2)

        working_dash_line = true_dash_line.copy()
        working_dash_line.move_to(working_net)

        def update_working_net(net):
            for node in net.nodes:
                node.set_stroke(color=get_cmap_color(0.5 + node.get_center()[1] - working_dash_line.get_center()[1]))
        working_net.add_updater(update_working_net)

        self.play(ShowCreation(working_dash_line))
        self.play(ShowCreation(working_net.nodes))
        self.bring_to_back(working_net.edges)
        self.play(FadeIn(working_net.edges))

        # begin using GrAPL

        grapl = GrAPL(G, 0.5, 1, alpha=1e-3)

        for t in range(15):

            # first pick is random
            if t == 0:
                j = np.random.choice(LINE_GRAPH_NUM_NODES)
            else:
                j = grapl.get_next_location()
            
            # go faster over time
            if t == 0:
                run_time = 1
            elif t < 5:
                run_time = 0.5
            else:
                run_time = 0.2

            to_highlight = working_net.nodes.submobjects[j].copy()
            to_highlight.set_color(YELLOW)
            self.bring_to_front(to_highlight)
            self.play(ShowCreationThenFadeOut(to_highlight), run_time=run_time)

            x = smooth_vals[j] + np.random.randn() * 0.2
            sampled_point = true_net.nodes[j].copy()
            sampled_point.set_color(get_cmap_color(x))
            sampled_point.shift([0, x - smooth_vals[j], 0])
            working_sampled_point = sampled_point.copy()
            working_sampled_point.shift([0, working_dash_line.get_center()[1] - true_dash_line.get_center()[1], 0])
            sampled_point_group = Group(sampled_point, working_sampled_point)
            self.bring_to_front(sampled_point_group)
            self.play(ShowCreation(sampled_point_group), run_time=run_time)
            self.play(sampled_point_group.fade, run_time=run_time)
            self.bring_to_back(sampled_point_group)

            grapl.update(j, x)
            net2 = self.create_line_graph_with_values(grapl.mu_hat)
            net2.shift(DOWN / 2 + working_dash_line.get_center())
            self.play(
                Transform(working_net.nodes, net2.nodes),
                Transform(working_net.edges, net2.edges),
                run_time=run_time
            )

class GrAPLvsOthersScene(Scene):

    def construct(self):

        n = 500
        num_rounds = 500

        p_inner = np.log(n // 2) / (n // 2)
        p_cross = p_inner / np.sqrt(n // 2)
        G = nx.stochastic_block_model([n // 2, n - n // 2], [[p_inner, p_cross], [p_cross, p_inner]])
        labels = np.ones(n)
        labels[n//2:] = -1
        component = max(nx.connected_components(G), key=len)
        G = G.subgraph(component).copy()
        labels = labels[list(component)]
        n = len(G)

        pos = nx.spring_layout(G)

        # separate the clusters a bit
        centroid1 = np.mean(np.stack([pos[node] for i, node in enumerate(G.nodes) if labels[i] == 1], 0), 0)
        centroid2 = np.mean(np.stack([pos[node] for i, node in enumerate(G.nodes) if labels[i] == -1], 0), 0)
        for i, node in enumerate(G.nodes):
            pos[node] += 0.3 * (centroid1 if labels[i] == 1 else centroid2)

        demo_net = Network(G, pos, node_scale=0.15*3/(2*MARGIN_SCALE), graph_scale=3)
        demo_net.center()

        self.play(ShowCreation(demo_net.nodes))
        self.add_foreground_mobject(demo_net.nodes)
        self.play(FadeIn(demo_net.edges))
        self.wait()

        # show true labels
        net2 = demo_net.copy_with_cvals((labels + 1) / 2)
        net2.center()
        self.play(Transform(demo_net.nodes, net2.nodes))
        self.wait()
        net2 = demo_net.copy_with_cvals()
        net2.center()
        self.play(Transform(demo_net.nodes, net2.nodes))
        self.wait()

        # split into three networks for GrAPL, random, and APT

        grapl_net = Network(G, pos, node_scale=0.1, graph_scale=1.5)
        random_net = grapl_net.copy_with_cvals()
        apt_net = grapl_net.copy_with_cvals()

        # hack: use the z axis to control color with our color map, and then control z with 3b1b transforms
        def update_nodes_color_from_z(nodes):
            for node in nodes.submobjects:
                node.set_stroke(color=get_cmap_color(node.get_z()))

        for net, name, color in zip([grapl_net, random_net, apt_net], ['GrAPL', 'Random', 'APT'], COLOR_CYCLE[:3]):
            net.color = color
            net.nodes.add_updater(update_nodes_color_from_z)
            net.title = TextMobject(name, color=color)
            net.title.scale(SMALL_SCALE)
            net.group = Group(net.title, net)
            net.group.arrange(direction=DOWN, center=True)
            net.nodes.set_z(0.5)

        # draw GrAPL net
        grapl_net.group.shift(2*UP + 3*LEFT)
        self.play(
            Transform(demo_net.nodes, grapl_net.nodes),
            Transform(demo_net.edges, grapl_net.edges)
        )
        self.remove(demo_net.nodes, demo_net.edges)
        self.bring_to_front(grapl_net.nodes)
        self.bring_to_back(grapl_net.edges)
        self.play(Write(grapl_net.title))

        # draw random net
        random_net.group.shift(2*DOWN + 3*LEFT)
        grapl_net_copy = grapl_net.copy()
        self.bring_to_back(grapl_net_copy.edges)
        self.play(
            Transform(grapl_net_copy.nodes, random_net.nodes),
            Transform(grapl_net_copy.edges, random_net.edges)
        )
        self.remove(grapl_net_copy.nodes, grapl_net_copy.edges)
        self.bring_to_front(random_net.nodes)
        self.bring_to_back(random_net.edges)
        self.play(Write(random_net.title))

        # draw APT net
        apt_net.group.shift(2*UP + 3*RIGHT)
        grapl_net_copy = grapl_net.copy()
        self.bring_to_back(grapl_net_copy.edges)
        self.play(
            Transform(grapl_net_copy.nodes, apt_net.nodes),
            Transform(grapl_net_copy.edges, apt_net.edges)
        )
        self.remove(grapl_net_copy.nodes, grapl_net_copy.edges)
        self.bring_to_front(apt_net.nodes)
        self.bring_to_back(apt_net.edges)
        self.play(Write(apt_net.title))

        # draw plot axes
        axes = MPLPlot(
            figsize=(3.5, 2.5), box=True,
            title='Error', title_scale=SMALL_SCALE,
            ticks=([0, num_rounds], [1, 0.1, 0.01, 0.001]), ticks_scale=0.7,
            xlims=(0, num_rounds), ylims=(0.00095, 1.05),
            yscale='log'
        )
        axes.shift(RIGHT * (apt_net.title.get_center() - axes.title.get_center()))
        axes.shift(UP * (random_net.title.get_center() - axes.title.get_center()))
        self.play(ShowCreation(axes))

        # initialize and play game
        grapl_net.grapl = GrAPL(G, 0, 100, lamda=1e-3, epsilon=1e-2, alpha=1)
        random_net.grapl = GrAPL(G, 0, 100, lamda=1e-3)
        apt_net.grapl = GrAPL(nx.empty_graph(n), 0, 1, lamda=1e-6, epsilon=1e-2, alpha=0)

        all_nets = [grapl_net, random_net, apt_net]
        for net in all_nets:
            net.errors = [1]
            net.mpl = axes.get_new_mpl_plot(color=net.color)

        fast_elapsed_time = 0
        fast_drawing_group = Group()
        use_fast_drawing_mode = False

        for t in range(num_rounds):

            if t % 10 == 0:
                print(t)

            if t < 2:
                run_time = 2 / (t + 1)
            else:
                run_time = 2 / 3 / (1 + 0.5 * num_rounds * (t - 3) * ((num_rounds - 1) - t) / ((num_rounds - 1) - 3)**2)
            
            if use_fast_drawing_mode and (run_time >= 3 / self.camera.frame_rate or fast_elapsed_time >= 1 / self.camera.frame_rate):
                self.wait(1 / self.camera.frame_rate)
                self.remove(fast_drawing_group)
                fast_elapsed_time -= 1 / self.camera.frame_rate
                fast_drawing_group = Group()

            if run_time < 3 / self.camera.frame_rate:
                use_fast_drawing_mode = True
                fast_elapsed_time += run_time
                self.bring_to_front(fast_drawing_group)
            else:
                use_fast_drawing_mode = False

            for net in all_nets:
                
                # first pick a random node
                if t == 0 or net is random_net:
                    j = np.random.choice(n)
                else:
                    j = net.grapl.get_next_location()
                
                net.to_highlight = net.nodes[j].copy()
                net.to_highlight.set_color(YELLOW)

                net.grapl.update(j, labels[j] + 2 * np.random.randn())
                net.labels = (np.sign(net.grapl.mu_hat) + 1) / 2
                if net is apt_net:
                    net.labels[net.grapl.n == 0] = 0.5

                # reminder: hack uses z values to determine color
                net.new_nodes = net.nodes.copy()
                for node, val in zip(net.new_nodes, net.labels):
                    node.set_z(val)
                
                net.errors.append(np.mean(2*net.labels - 1 != labels))
                net.mpl.add_curve([t, t+1], net.errors[-2:])

            if use_fast_drawing_mode:
                fast_drawing_group.add(*(net.to_highlight for net in all_nets))
                for net in all_nets:
                    for node, new_node in zip(net.nodes, net.new_nodes):
                        node.move_to(new_node)
                    net.mpl.animate_grow_last()
                    self.add(net.mpl.curves[-1])
            else:
                self.bring_to_front(*(net.to_highlight for net in all_nets))
                self.play(
                    AnimationGroup(*(Transform(net.nodes, net.new_nodes, run_time=run_time, rate_func=squish_rate_func(smooth, 0.5, 1)) for net in all_nets), run_time=run_time),
                    AnimationGroup(*(VFadeInThenOut(net.to_highlight, run_time=run_time, rate_func=squish_rate_func(there_and_back, 0, 0.5)) for net in all_nets), run_time=run_time),
                    AnimationGroup(*(net.mpl.animate_grow_last(run_time=run_time, rate_func=linear) for net in all_nets), run_time=run_time)
                )
            

