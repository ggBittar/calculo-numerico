from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(10, 7), tight_layout=True, facecolor="#202124")
        self.ax_lines = self.figure.add_subplot(211)
        self.ax_conv = self.figure.add_subplot(212)
        self._line_data = []
        self._conv_data = []
        self._labels = ("x", "x")
        super().__init__(self.figure)
        self._style_axes(self.ax_lines, self.ax_conv)

    def _style_axes(self, *axes):
        for ax in axes:
            ax.set_facecolor("#171717")
            ax.tick_params(colors="#e8eaed")
            for spine in ax.spines.values():
                spine.set_color("#9aa0a6")
            ax.xaxis.label.set_color("#e8eaed")
            ax.yaxis.label.set_color("#e8eaed")
            ax.title.set_color("#e8eaed")

    def redraw(self, line_data, conv_data, diff_var: str, plot_var: str):
        self._line_data = list(line_data)
        self._conv_data = list(conv_data)
        self._labels = (diff_var, plot_var)
        self.ax_lines.clear()
        self.ax_conv.clear()
        self._style_axes(self.ax_lines, self.ax_conv)
        self._draw_lines(self.ax_lines)
        self._draw_conv(self.ax_conv)
        self.draw_idle()

    def _draw_lines(self, ax):
        diff_var, plot_var = self._labels
        for label, x, y in self._line_data:
            ax.plot(x, y, label=label)
        ax.set_title(f"Estimativas de d/d{diff_var} ao longo de {plot_var}")
        ax.set_xlabel(plot_var)
        ax.set_ylabel(f"d/d{diff_var} f")
        ax.grid(True, alpha=0.25)
        if self._line_data:
            ax.legend(fontsize=8)

    def _draw_conv(self, ax):
        for method_name, n_vals, err_vals in self._conv_data:
            ax.plot(n_vals, err_vals, marker="o", label=method_name)
        ax.set_title("Convergência da malha (erro máximo na linha analisada)")
        ax.set_xlabel("Número de elementos")
        ax.set_ylabel("Erro máximo absoluto")
        ax.grid(True, alpha=0.25)
        if self._conv_data:
            ax.legend(fontsize=8)

    def export_line_plot(self, filepath: str, width: int = 1920, height: int = 1080):
        fig = Figure(figsize=(width / 100, height / 100), dpi=100, tight_layout=True, facecolor="#202124")
        ax = fig.add_subplot(111)
        self._style_axes(ax)
        self._draw_lines(ax)
        fig.savefig(filepath, dpi=100, facecolor=fig.get_facecolor())

    def export_convergence_plot(self, filepath: str, width: int = 1920, height: int = 1080):
        fig = Figure(figsize=(width / 100, height / 100), dpi=100, tight_layout=True, facecolor="#202124")
        ax = fig.add_subplot(111)
        self._style_axes(ax)
        self._draw_conv(ax)
        fig.savefig(filepath, dpi=100, facecolor=fig.get_facecolor())
