"""Manim Community scenes for ML Academy lessons (3Blue1Brown-style).

No LaTeX is installed on this machine, so every label uses Text, never MathTex/Tex.
Render:  python animations/render.py          (writes src/web/public/manim/<Scene>.mp4)
"""
from manim import *
import numpy as np

BLUE_ = "#2563eb"
ORANGE_ = "#d97706"


class LinearMap(Scene):
    """f2n.03: a matrix is a linear map; its columns are where the basis vectors land."""

    def construct(self):
        plane = NumberPlane(background_line_style={"stroke_opacity": 0.35})
        title = Text("A matrix moves the whole plane", font_size=30).to_edge(UP)
        i_hat = Arrow(ORIGIN, RIGHT, buff=0, color=GREEN)
        j_hat = Arrow(ORIGIN, UP, buff=0, color=RED)
        self.play(Create(plane), Write(title))
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        self.wait(0.5)
        m = np.array([[2, 1], [0, 1.5]])
        cap = Text("columns = where i and j land", font_size=24, color=YELLOW).to_edge(DOWN)
        self.play(
            plane.animate.apply_matrix(m),
            i_hat.animate.put_start_and_end_on(ORIGIN, np.append(m[:, 0], 0)),
            j_hat.animate.put_start_and_end_on(ORIGIN, np.append(m[:, 1], 0)),
            Write(cap),
            run_time=3,
        )
        self.wait(1.5)


class Eigenvectors(Scene):
    """math.07: most vectors turn under a map; eigenvectors only stretch."""

    def construct(self):
        m = np.array([[3, 1], [0, 2]])
        plane = NumberPlane(background_line_style={"stroke_opacity": 0.3})
        title = Text("Eigenvectors stay on their line", font_size=30).to_edge(UP)
        self.play(Create(plane), Write(title))
        eig1 = Arrow(ORIGIN, RIGHT * 1.0, buff=0, color=ORANGE_)          # eigenvalue 3
        eig2 = Arrow(ORIGIN, np.array([-1.0, 1.0, 0]), buff=0, color=GREEN)  # eigenvalue 2
        other = Arrow(ORIGIN, np.array([0.5, 1.5, 0]), buff=0, color=BLUE_)
        lines = VGroup(
            Line(LEFT * 7, RIGHT * 7, color=ORANGE_, stroke_opacity=0.5),
            Line(np.array([-5, 5, 0]), np.array([5, -5, 0]), color=GREEN, stroke_opacity=0.5),
        )
        self.play(Create(lines), *[GrowArrow(a) for a in (eig1, eig2, other)])
        lab = Text("blue turns, orange and green only scale", font_size=22, color=YELLOW).to_edge(DOWN)
        self.play(
            plane.animate.apply_matrix(m),
            lines.animate.apply_matrix(m),
            eig1.animate.put_start_and_end_on(ORIGIN, np.append(m @ [1.0, 0], 0)),
            eig2.animate.put_start_and_end_on(ORIGIN, np.append(m @ [-1.0, 1.0], 0)),
            other.animate.put_start_and_end_on(ORIGIN, np.append(m @ [0.5, 1.5], 0)),
            Write(lab),
            run_time=3.5,
        )
        self.wait(1.5)


class GradientDescent(Scene):
    """ml.07: follow the negative gradient downhill on a loss surface (level curves)."""

    def construct(self):
        f = lambda x, y: 0.5 * (x**2 + 6 * y**2)
        grad = lambda x, y: np.array([x, 6 * y])
        ax = Axes(x_range=[-4, 4], y_range=[-2.5, 2.5], x_length=9, y_length=5.6, tips=False)
        title = Text("Gradient descent: step against the gradient", font_size=28).to_edge(UP)
        contours = VGroup(*[
            ParametricFunction(
                lambda t, c=c: ax.c2p(np.sqrt(2 * c) * np.cos(t), np.sqrt(2 * c / 6) * np.sin(t)),
                t_range=[0, TAU], color=BLUE_, stroke_opacity=0.6)
            for c in (0.3, 1, 2, 3.5, 5.5, 8)
        ])
        self.play(Create(ax), Write(title))
        self.play(Create(contours, lag_ratio=0.15))
        p = np.array([-3.4, 1.9])
        dot = Dot(ax.c2p(*p), color=ORANGE_)
        self.play(FadeIn(dot))
        lr = 0.28
        for _ in range(14):
            g = grad(*p)
            new = p - lr * g
            arrow = Arrow(ax.c2p(*p), ax.c2p(*new), buff=0, color=RED, stroke_width=4, max_tip_length_to_length_ratio=0.35)
            seg = Line(ax.c2p(*p), ax.c2p(*new), color=ORANGE_)
            self.play(GrowArrow(arrow), run_time=0.25)
            self.play(dot.animate.move_to(ax.c2p(*new)), FadeOut(arrow), run_time=0.2)
            self.add(seg)
            p = new
        end = Text("loss goes down at each step", font_size=24, color=YELLOW).to_edge(DOWN)
        self.play(Write(end))
        self.wait(1.5)


class Derivative(Scene):
    """f2n.04: the derivative is the slope of the tangent, the limit of secants."""

    def construct(self):
        ax = Axes(x_range=[-0.5, 4], y_range=[-0.5, 9], x_length=8, y_length=5, tips=False)
        graph = ax.plot(lambda x: x**2 / 2 + 0.5, color=BLUE_)
        title = Text("Secant slopes converge to the tangent slope", font_size=28).to_edge(UP)
        self.play(Create(ax), Write(title))
        self.play(Create(graph))
        x0 = 1.5
        h = ValueTracker(2.0)
        f = lambda x: x**2 / 2 + 0.5
        sec = always_redraw(lambda: ax.get_secant_slope_group(
            x0, graph, dx=h.get_value(), secant_line_color=ORANGE_, secant_line_length=7))
        dot = Dot(ax.c2p(x0, f(x0)), color=RED)
        self.play(FadeIn(dot), Create(sec))
        self.play(h.animate.set_value(0.01), run_time=4, rate_func=smooth)
        lab = Text("slope at x = 1.5 is 1.5", font_size=24, color=YELLOW).to_edge(DOWN)
        self.play(Write(lab))
        self.wait(1.5)
