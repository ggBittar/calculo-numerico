from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QTabWidget

from .tabs.derivative_tab import DerivativeTab
from .tabs.pde_tab import PdeTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PR1 - Estimativas Numéricas")
        self.resize(1280, 860)

        tabs = QTabWidget()
        tabs.addTab(DerivativeTab(), "Estimativa de Derivadas")
        tabs.addTab(PdeTab(), "Estimativa via EDP")

        self.setCentralWidget(tabs)
