import sys
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QLabel
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

    
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

    
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Comparação de Diferenças Finitas")
        self.resize(1100, 800)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        nx = 5
        L = 1
        x0 = 0
        x = np.linspace(x0, L, nx)
        dx = x[1] - x[0]

        alpha = 0.01

        t0 = 0
        tf = 10
        
        nt = 100
        t = np.linspace(t0, tf, nt)
        dt = t[1] - t[0]

        T = np.zeros((nx, nt))
        T[:, 0] = 300*np.ones(nx)
        T[0, :] = 273*np.ones(nt)
        T[-1, :] = 373*np.ones(nt)

        for n in range(0, nt-1):
            for i in range(1, nx-1):
                T[i, n+1] = T[i, n] + (alpha*dt/(dx**2)) * (T[i+1, n] - 2*T[i, n] + T[i-1, n])
            plotTab = PlotTab(x, T[:, n+1], t[n+1])
            tabs.addTab(plotTab, f"t={t[n+1]:.2f}")
        # Aba final com tabela de erros


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())