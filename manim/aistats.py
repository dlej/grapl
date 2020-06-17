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
MED_SMALL_SCALE = 0.8
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
        title_vg.arrange(direction=DOWN)
        title_vg.shift(1.5*UP)

        self.play(Write(title_vg))
        self.wait()

        authors = [
            ['Daniel LeJeune (speaker)', 'daniel.jpg', 'Rice University'],
            ['Gautam Dasarathy', 'gautam.jpg', 'Arizona State University'],
            ['Richard G. Baraniuk', 'baraniuk.jpg', 'Rice University']
        ]

        photos = Group()
        author_affils = VGroup()

        for i, (author_name, photo_filename, university) in enumerate(authors):

            photo = ImageMobject(os.path.join(PATH_TO_IMAGES, photo_filename))
            photo.scale(MARGIN_SCALE)
            photo.move_to([-4 + 4*i, -1, 0], aligned_edge=ORIGIN)
            photos.add(photo)

            name_mobject = TextMobject(author_name)
            univ_mobject = TextMobject(university)
            univ_mobject.scale(MED_SMALL_SCALE)

            author_affil_vg = VGroup(name_mobject, univ_mobject)
            author_affil_vg.scale(SMALL_SCALE)
            author_affil_vg.arrange(direction=DOWN, aligned_edge=ORIGIN)
            # fix alignment of affiliation with dangling letters:
            author_affil_vg.next_to(photo, direction=DOWN)
            if i > 0:
                univ_mobject.move_to(author_affils[0][-1], coor_mask=UP)
            author_affils.add(author_affil_vg)

        self.play(
            LaggedStartMap(
                lambda x: LaggedStartMap(FadeIn, x, lag_ratio=0.1),
                Group(photos, author_affils),
                lag_ratio=0.5
            )
        )
        self.wait()

        self.play(
            FadeOut(title_vg),
            LaggedStartMap(
                lambda x: LaggedStartMap(FadeOut, x, lag_ratio=0.1),
                Group(*[Group(*z) for z in zip(photos, author_affils)]),
                lag_ratio=0.1
            )
        )
        self.wait()

class InformalIntroScene(Scene):

    def construct(self):

        title = TextMobject('Thresholding Bandits', substrings_to_isolate=['Thresholding', 'Bandits'])
        title.move_to(3*UP)

        self.play(Write(title))
        self.wait()

        positions = [
            2*UP + 1.7*LEFT,
            1.8*UP + 2*RIGHT,
            0.3*UP + 3.7*LEFT,
            0.2*DOWN + 0.1*RIGHT,
            3.5*RIGHT,
            2*DOWN + 2*LEFT,
            2.1*DOWN + 1.7*RIGHT
        ]
        mus = [
            1.2,
            0.5,
            1.1,
            0.8,
            0.3,
            1.3,
            0.6
        ]
        bandits = Group()
        bandit_mus = Group()

        for i, (pos, mu) in enumerate(zip(positions, mus)):
            bandit = ImageMobject(os.path.join(PATH_TO_IMAGES, 'slotmachine.png'))
            bandit.scale(0.5)
            bandit.move_to(pos)
            mu_label = TextMobject(r'$\mu_{%d} = %g$' % (i, mu))
            mu_label.scale(0.5)
            mu_label.next_to(bandit, DOWN)
            bandit_mus.add(mu_label)
            bandits.add(bandit)
        
        self.play(
            LaggedStartMap(FadeIn, bandits, lag_ratio=0.05),
            LaggedStartMap(FadeIn, bandit_mus, lag_ratio=0.05)
        )
        self.wait()

        k_greater_than_1 = TextMobject(r'$\{k : \mu_k > 1\} = \, ?$')
        k_greater_than_1.scale(0.7)
        k_greater_than_1.next_to(title, DOWN)
        k_greater_than_1.shift(4.5*RIGHT)

        self.play(Write(k_greater_than_1))
        self.wait()

        n_trials = 14
        np.random.seed(42)
        sample_order = np.random.permutation(7)
        for i in range(n_trials):
            j = sample_order[i % 7]
            x = np.random.exponential(mus[j])
            x_label = TextMobject(r'$x_{%d} = %.2f$' % (i, x), color=get_cmap_color((x - mus[j]) + 0.5))
            x_label.scale(0.5)
            x_label.next_to(bandit_mus[j], DOWN, buff=SMALL_BUFF)
            circle = Circle(stroke_color=YELLOW)
            circle.scale(0.65)
            circle.move_to(bandits[j])
            self.play(
                LaggedStartMap(ShowCreationThenFadeOut, Group(circle, x_label), lag_ratio=0.5)
            )
        self.wait()

        edge_connections = [
            (0, 2),
            (0, 3),
            (1, 3),
            (1, 6),
            (1, 4),
            (2, 5),
            (3, 5),
            (3, 6)
        ]

        edges = VGroup()

        for i, j in edge_connections:
            edge = DashedLine(positions[i], positions[j])
            edges.add(edge)

        thresholding_tex_old = title.get_part_by_tex('Thresholding')
        bandits_tex_old = title.get_part_by_tex('Bandits')

        title_graph = TextMobject('Thresholding Graph Bandits', substrings_to_isolate=['Thresholding', 'Graph', 'Bandits'])
        thresholding_tex_new = title_graph.get_part_by_tex('Thresholding')
        bandits_tex_new = title_graph.get_part_by_tex('Bandits')
        graph = title_graph.get_part_by_tex('Graph')
        graph.set_color(YELLOW)

        title_graph.move_to(title)

        self.bring_to_back(edges)
        self.play(
            AnimationGroup(
                AnimationGroup(
                    ApplyMethod(thresholding_tex_old.move_to, thresholding_tex_new),
                    ApplyMethod(bandits_tex_old.move_to, bandits_tex_new),
                    ShowCreation(edges),
                    ApplyFunction(lambda x: x, bandits), # keep bandits on top
                    ApplyFunction(lambda x: x, bandit_mus), # keep bandits on top
                    lag_ratio=0
                ),
                Write(graph),
                lag_ratio=0.5
            )
        )
        self.wait()

        to_mark = [0, 2, 5]
        bandits_to_mark = Group(*sum(([bandits[i], bandit_mus[i]] for i in to_mark), []))
        blue_rect = Rectangle(width=bandits_to_mark.get_width() + 0.3, height=bandits_to_mark.get_height() + 0.3, color=BLUE)
        blue_rect.move_to(bandits_to_mark)

        self.play(
            ShowCreation(blue_rect),
            k_greater_than_1.set_color, BLUE
        )
        self.wait()

class FormalIntroScene(Scene):

    def construct(self):

        arms = TextMobject(r'$\nu_i : i \in [N]$')
        arms.move_to(2.3*UP)
        self.play(Write(arms))
        self.wait()

        arms_stats = TextMobject(r'$\mathbb{E}_{X \sim \nu_i}[X] = \mu_i$, $R$-sub-Gaussian')
        arms_stats.next_to(arms, DOWN)
        self.play(Write(arms_stats))
        self.wait()

        superlevelset = TextMobject(r'$\mathcal{S}_\tau = \{i : \mu_i \geq \tau\}$')
        superlevelset.next_to(arms_stats, DOWN)
        self.play(Write(superlevelset))
        self.wait()

        error = TextMobject(r'$\widehat{\mathcal{S}} = \mathcal{S}_\tau$ for $i : |\mu_i - \tau| > \varepsilon$?')
        error.next_to(superlevelset, DOWN)
        self.play(Write(error))
        self.wait()

        policy = TextMobject(r'$(\pi_t)_{t=1}^T, \, \pi_t \in [N]$')
        policy.next_to(error, DOWN)
        self.play(Write(policy))
        self.wait()

        sample = TextMobject(r'$x_t \sim \nu_{\pi_t}$')
        sample.next_to(policy, DOWN)
        self.play(Write(sample))
        self.wait()
        
        graph = TextMobject(r'''$
            (\mathcal{V}, \mathcal{E}, \mathbf{W}) \to \mathbf{L}
        $''')
        graph.next_to(sample, DOWN)
        self.play(Write(graph))
        self.wait()

        to_fade = Group(
            arms, arms_stats, superlevelset, error, policy, sample, graph
        )
        self.play(FadeOut(to_fade))
        self.wait()

class AlgorithmScene(Scene):

    def construct(self):
        
        grapl_parts = ['Gr', 'A', 'P', 'L']
        grapl = TextMobject('GrAPL', substrings_to_isolate=grapl_parts)
        grapl.arrange(RIGHT, coor_mask=RIGHT, buff=0.03)
        grapl.move_to(3*UP)
        self.play(Write(grapl))
        self.wait()

        grapl_expanded = TextMobject('Graph-based Anytime Parameter-Light thresholding algorithm', substrings_to_isolate=grapl_parts)
        # reorganize into words
        grapl_reorganized = VGroup(grapl_expanded[:2], grapl_expanded[2:4], grapl_expanded[4:8])
        for word in grapl_reorganized:
            word.arrange(RIGHT, buff=0.03, coor_mask=RIGHT, center=False)
        grapl_reorganized.arrange(RIGHT, buff=0.2, coor_mask=RIGHT)
        grapl_expanded.scale(0.7)
        grapl_expanded.move_to(grapl)

        # acronym components are even indices
        grapl_expanded_acronym = VGroup(*[grapl_expanded[i] for i in range(0, len(grapl_expanded), 2)])
        grapl_expanded_acronym.set_color(YELLOW)
        grapl_expanded_filling = VGroup(*[grapl_expanded[i] for i in range(1, len(grapl_expanded), 2)])

        grapl_copy = grapl.copy()
        self.play(
            AnimationGroup(
                AnimationGroup(*[
                    Transform(a, b) for a, b in zip(grapl, grapl_expanded_acronym)
                ]),
                FadeIn(grapl_expanded_filling),
                lag_ratio=0.5
            )
        )
        self.wait()

        apt_plus_graph = TextMobject('APT + Laplacian regularization')
        apt_plus_graph.scale(0.7)
        apt_plus_graph.next_to(grapl_expanded, DOWN)
        self.play(Write(apt_plus_graph))
        self.wait()

        self.play(
            AnimationGroup(
                FadeOut(apt_plus_graph),
                FadeOut(grapl_expanded_filling),
                AnimationGroup(*[
                    Transform(a, b) for a, b in zip(grapl, grapl_copy)
                ]),
                lag_ratio=0.1
            )
        )
        self.wait()

        step_one = TextMobject('1. Estimate distances from threshold', color=BLUE)
        step_one.move_to(2*UP)

        delta_eq = TextMobject(r'''$
            \widehat{\Delta}_i^t = |\widehat{\mu}_i^t - \tau| + \varepsilon
        $''')
        delta_eq.next_to(step_one, DOWN)

        self.play(Write(step_one))
        self.play(Write(delta_eq))
        self.wait()

        epsilon = TextMobject(r'$\varepsilon$', color=YELLOW)
        epsilon.move_to(delta_eq)
        epsilon.shift(1.745*RIGHT + 0.075*DOWN)

        self.play(FadeIn(epsilon))
        self.wait()
        self.play(FadeOut(epsilon))
        self.wait()

        step_two = TextMobject('2. Compute confidence proxies', color=BLUE)

        z_eq = TextMobject(r'''$
            z_i^t = \widehat{\Delta}_i^t \sqrt{n_i^t + \alpha}
        $''')
        z_eq.next_to(step_two, DOWN)

        self.play(Write(step_two))
        self.play(Write(z_eq))
        self.wait()

        n = TextMobject(r'$n_i^t$', color=YELLOW)
        n.move_to(z_eq)
        n.shift(0.6*RIGHT + 0.045*DOWN)

        self.play(FadeIn(n))
        self.wait()
        self.play(FadeOut(n))
        self.wait()

        alpha = TextMobject(r'$\alpha$', color=YELLOW)
        alpha.move_to(z_eq)
        alpha.shift(1.625*RIGHT + 0.065*DOWN)

        self.play(FadeIn(alpha))
        self.wait()
        self.play(FadeOut(alpha))
        self.wait()

        step_three = TextMobject('3. Select next arm', color=BLUE)
        step_three.move_to(2*DOWN)

        pi_eq = TextMobject(r'''$
            \pi_{t+1} = \underset{i \in [N]}{\mathrm{arg min}} \, z_i^t
        $''')
        pi_eq.next_to(step_three, DOWN)

        self.play(Write(step_three))
        self.play(Write(pi_eq))
        self.wait()

        mu = TextMobject(r'$\widehat{\mu}^t$', color=YELLOW)
        mu.move_to(delta_eq)
        mu.shift(0.25*LEFT + 0.04*DOWN)

        self.play(FadeIn(mu))
        self.wait()

        to_fade = Group(step_one, delta_eq, step_two, z_eq, step_three, pi_eq)
        self.play(
            AnimationGroup(
                FadeOut(to_fade),
                ApplyFunction(lambda x: x, mu), # keep on top
                lag_ratio=0
            )
        )

        mu_eq = TextMobject(r'''$
            \widehat{\boldsymbol{\mu}}^t 
            = \underset{\boldsymbol{\mu}}{\mathrm{arg min}} \, 
            \sum_{s = 1}^t (x_s - \mu_{\pi_s})^2
            + \gamma \boldsymbol{\mu}^\top \mathbf{L}_\lambda \boldsymbol{\mu}
        $''')
        bold_mu = TextMobject(r'$\widehat{\boldsymbol{\mu}}^t$', color=YELLOW)
        bold_mu.move_to(mu_eq, aligned_edge=LEFT)
        bold_mu.shift(0.175*UP)

        self.play(Transform(mu, bold_mu))
        self.play(Write(mu_eq))
        self.remove(mu)
        self.wait()

        laplacian_norm = TextMobject(r'''$
            \boldsymbol{\mu}^\top \mathbf{L}_\lambda \boldsymbol{\mu}
        $''', color=YELLOW)
        laplacian_norm.move_to(mu_eq, aligned_edge=RIGHT)
        laplacian_norm.shift(0.16*UP)

        laplacian_eq = TextMobject(r'''$
            \boldsymbol{\mu}^\top \mathbf{L}_\lambda \boldsymbol{\mu}
            = \sum_{(i, j) \in \mathcal{E}} w_{ij} (\mu_i - \mu_j)^2 + \lambda || \boldsymbol{\mu} ||_2^2
        $''')
        laplacian_eq.next_to(mu_eq, DOWN)
        laplacian_norm_copy = laplacian_norm.copy()
        laplacian_norm_copy.move_to(laplacian_eq, aligned_edge=LEFT)
        laplacian_norm_copy.shift(0.07*UP)

        self.play(FadeIn(laplacian_norm))
        self.wait()
        
        self.play(Transform(laplacian_norm, laplacian_norm_copy))
        self.play(Write(laplacian_eq))
        self.remove(laplacian_norm)
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
        self.wait()

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
        self.wait()

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
            elif t == 14:
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
        
        self.wait()

class TheoremScene(Scene):

    def construct(self):

        theorem = TextMobject(r'\textbf{Theorem 1.}', color=BLUE)
        theorem.shift(2*UP + 4*LEFT)

        iff = TextMobject(r'If', color=YELLOW)
        iff.next_to(theorem, DOWN, coor_mask=UP)
        iff.move_to(theorem, aligned_edge=LEFT, coor_mask=RIGHT)
        iff.shift(0.5*RIGHT)

        self.play(Write(theorem))
        self.wait()
        self.play(Write(iff))
        self.wait()

        conditions_tex = [
            r'''$
                T \geq C \gamma \boldsymbol{\mu}^\top \mathbf{L}_\lambda \boldsymbol{\mu}
            $'''
        ]
        conditions = []
        for tex in conditions_tex:
            condition = TextMobject(tex)
            prev_obj = iff if len(conditions) == 0 else conditions[-1]
            condition.next_to(prev_obj, DOWN, coor_mask=UP)
            condition.move_to(iff, aligned_edge=LEFT, coor_mask=RIGHT)
            condition.shift(0.5*RIGHT)
            conditions.append(condition)

        for condition in conditions:
            self.play(Write(condition))
            self.wait()

        thenn = TextMobject('then', color=YELLOW)
        thenn.next_to(conditions[-1], DOWN)
        thenn.move_to(iff, aligned_edge=LEFT, coor_mask=RIGHT)

        self.play(Write(thenn))
        self.wait()

        result_first = TextMobject(r'''$
            \mathrm{error} \leq \exp \Big\{ - C' \frac{\gamma}{R^2} T % \Big\} # hack
        $''')
        result_first.next_to(thenn, DOWN)
        result_first.move_to(thenn, aligned_edge=LEFT, coor_mask=RIGHT)
        result_first.shift(0.5*RIGHT)

        result_dimlog = TextMobject(r'${}+ d_T \log (1 + \frac{T}{\gamma \lambda})$')
        result_dimlog.move_to(result_first, aligned_edge=LEFT)
        result_dimlog.shift(5.135*RIGHT)

        result_brace = TextMobject(r'$\Big\{ \hspace{30em} \Big \}$') # hack; can't put just one brace, so put one super far to the left
        result_brace.move_to(result_dimlog, aligned_edge=RIGHT)
        result_brace.shift(0.35*RIGHT)

        result = VGroup(result_first, result_dimlog, result_brace)
        
        self.play(Write(result))
        self.wait()

        dimension = TextMobject(r'$d_T$', color=YELLOW)
        dimension.move_to(result_dimlog)
        dimension.shift(1.115*LEFT + 0.01*UP)

        self.play(FadeIn(dimension))

        dimension_lt_n = TextMobject(r'$d_T \ll N$')
        dimension_lt_n.next_to(dimension, DOWN, buff=MED_SMALL_BUFF)
        dimension_copy = dimension.copy()
        dimension_copy.move_to(dimension_lt_n, aligned_edge=LEFT)

        self.play(dimension.move_to, dimension_copy)
        self.play(Write(dimension_lt_n))
        self.remove(dimension)
        self.wait()

        t_gt_dim = TextMobject(r'$T \gtrsim d_T$')
        t_gt_dim.next_to(condition, RIGHT, buff=MED_LARGE_BUFF)
        t_gt_dim.shift(0.035*DOWN)

        self.play(Write(t_gt_dim))
        self.wait()

        self.play(
            AnimationGroup(
                AnimationGroup(
                    FadeOut(result_dimlog),
                    FadeOut(dimension_lt_n),
                    lag_ratio=0
                ),
                ApplyMethod(result_brace.shift, (result_dimlog.get_width() + 0.05)*LEFT),
                lag_ratio=0.1
            )
        )
        self.wait()

class GrAPLvsOthersScene(Scene):
    '''warning: takes an hour to compile at 1080p on my machine
    '''

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
        self.wait()

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
        self.wait()

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
        self.wait()

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
        self.play(ShowCreation(axes), run_time=2)
        self.wait()

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
        self.wait()
            
class AcknowledgmentScene(Scene):

    def construct(self):

        arxiv = TextMobject('arXiv:1905.09190', color=BLUE)
        arxiv.shift(2*UP + 3*LEFT)

        self.play(Write(arxiv))
        self.wait()

        points_tex = [
            'lower bounds',
            'optimality and tuning',
            'experiments'
        ]

        points = []
        for tex in points_tex:
            point = TextMobject(r'$\cdot$ ' + tex)
            prev_obj = arxiv if len(points) == 0 else points[-1]
            point.next_to(prev_obj, DOWN, coor_mask=UP)
            point.move_to(arxiv, aligned_edge=LEFT, coor_mask=RIGHT)
            point.shift(0.5*RIGHT)
            points.append(point)

        for point in points:
            self.play(Write(point))
            self.wait()

        ack_text = '''
        This work was supported by NSF grants CCF-1911094, IIS-1838177, and IIS1730574; 
        ONR grants N00014-18-12571 and N00014- 17-1-2551; AFOSR grant FA9550-18-1-0478; 
        DARPA grant G001534-7500; and a Vannevar Bush Faculty Fellowship, ONR grant N00014-18-1-2047.
        '''
        ack = TextMobject(ack_text)
        ack.scale(0.7)
        ack.next_to(points[-1], DOWN, buff=LARGE_BUFF, coor_mask=UP)

        self.play(FadeIn(ack))
        self.wait()