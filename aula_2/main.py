import sys
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QLabel
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from metodos.diferencas_finitas import (
    f, df,
    finite_difference_central,
    finite_difference_forward,
    finite_difference_backward
)


class PlotTab(QWidget):
    def __init__(self, x, dfx, dfx_central, dfx_forward, dfx_backward, n):
        super().__init__()

        layout = QVBoxLayout(self)

        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.plot(x, dfx, label="Derivada exata", color="black")
        ax.plot(x[1:-1], dfx_central, label="Diferença central", linestyle="--")
        ax.plot(x[:-1], dfx_forward, label="Diferença progressiva", linestyle="--")
        ax.plot(x[1:], dfx_backward, label="Diferença regressiva", linestyle="--")

        ax.set_title(f"Derivada de cos(x) com n={n} pontos")
        ax.set_xlabel("x")
        ax.set_ylabel("f'(x)")
        ax.legend()
        ax.grid(True)

        layout.addWidget(canvas)


class ErrorTableTab(QWidget):
    def __init__(self, error_data):
        super().__init__()

        layout = QVBoxLayout(self)

        label = QLabel("Erros absolutos no ponto de índice n//3 do domínio")
        layout.addWidget(label)

        table = QTableWidget()
        table.setRowCount(len(error_data))
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "n",
            "x[n//3]",
            "Erro central",
            "Erro progressiva",
            "Erro regressiva"
        ])

        for row, item in enumerate(error_data):
            n, x_eval, err_c, err_f, err_b = item

            table.setItem(row, 0, QTableWidgetItem(str(n)))
            table.setItem(row, 1, QTableWidgetItem(f"{x_eval:.8f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{err_c:.8e}"))
            table.setItem(row, 3, QTableWidgetItem(f"{err_f:.8e}"))
            table.setItem(row, 4, QTableWidgetItem(f"{err_b:.8e}"))

        table.resizeColumnsToContents()
        layout.addWidget(table)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Comparação de Diferenças Finitas")
        self.resize(1100, 800)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        L = 4 * np.pi
        xo = -2 * np.pi
        nx = [13, 25, 49, 97, 193]

        error_data = []

        for n in nx:
            x = np.linspace(xo, xo + L, n)
            dx = x[1] - x[0]

            fx = f(x)
            dfx = df(x)

            dfx_central = finite_difference_central(fx, dx)
            dfx_forward = finite_difference_forward(fx, dx)
            dfx_backward = finite_difference_backward(fx, dx)

            # Aba do gráfico
            plot_tab = PlotTab(
                x, dfx, dfx_central, dfx_forward, dfx_backward, n
            )
            tabs.addTab(plot_tab, f"Gráfico n={n}")

            # Ponto de análise: índice n//3
            i = (n-1) // 3 
            x_eval = x[i]
            dfx_exact = dfx[i]

            # Alinhamento dos vetores numéricos:
            # central  -> definido em x[1:-1], então índice correspondente é i-1
            # forward  -> definido em x[:-1], então índice correspondente é i
            # backward -> definido em x[1:], então índice correspondente é i-1

            err_c = np.nan
            err_f = np.nan
            err_b = np.nan

            if 1 <= i <= n - 2:
                err_c = abs(dfx_central[i + 1] - dfx_exact)

            if 0 <= i <= n - 2:
                err_f = abs(dfx_forward[i] - dfx_exact)

            if 1 <= i <= n - 1:
                err_b = abs(dfx_backward[i + 1] - dfx_exact)

            error_data.append((n, x_eval, err_c, err_f, err_b))

        # Aba final com tabela de erros
        error_tab = ErrorTableTab(error_data)
        tabs.addTab(error_tab, "Tabela de erros")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())