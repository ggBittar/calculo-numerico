from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(10, 7), tight_layout=True, facecolor="#202124")
        self.ax_lines = self.figure.add_subplot(211)
        self.ax_conv = self.figure.add_subplot(212)
        super().__init__(self.figure)
        for ax in [self.ax_lines, self.ax_conv]:
            ax.set_facecolor("#171717")
            ax.tick_params(colors="#e8eaed")
            for spine in ax.spines.values():
                spine.set_color("#9aa0a6")
            ax.xaxis.label.set_color("#e8eaed")
            ax.yaxis.label.set_color("#e8eaed")
            ax.title.set_color("#e8eaed")

    def redraw(self, line_data, conv_data, diff_var: str, plot_var: str):
        self.ax_lines.clear()
        self.ax_conv.clear()
        for ax in [self.ax_lines, self.ax_conv]:
            ax.set_facecolor("#171717")
            ax.tick_params(colors="#e8eaed")
            for spine in ax.spines.values():
                spine.set_color("#9aa0a6")
        for label, x, y in line_data:
            self.ax_lines.plot(x, y, label=label)
        self.ax_lines.set_title(f"Estimativas de d/d{diff_var} ao longo de {plot_var}")
        self.ax_lines.set_xlabel(plot_var)
        self.ax_lines.set_ylabel(f"d/d{diff_var} f")
        self.ax_lines.grid(True, alpha=0.25)
        if line_data:
            self.ax_lines.legend(fontsize=8)

        for method_name, n_vals, err_vals in conv_data:
            self.ax_conv.plot(n_vals, err_vals, marker="o", label=method_name)
        self.ax_conv.set_title("Convergência da malha (erro máximo na linha analisada)")
        self.ax_conv.set_xlabel("Número de elementos")
        self.ax_conv.set_ylabel("Erro máximo absoluto")
        self.ax_conv.grid(True, alpha=0.25)
        if conv_data:
            self.ax_conv.legend(fontsize=8)
        self.draw_idle()
