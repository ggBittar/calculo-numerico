from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class PlotTab(QWidget):
    def __init__(self, x, y, t):
        super().__init__()

        layout = QVBoxLayout(self)

        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.plot(x, y, label=f"t={t}", linestyle="--")

        ax.set_title(f"instante t={t}")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.legend()
        ax.grid(True)

        layout.addWidget(canvas)
