from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit
from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton
from PyQt6.QtWidgets import QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView

from analysis_core import run_analysis
from expression_utils import VARIABLES, parse_n_values
from method_registry import MethodRegistry
from plot_widget import PlotCanvas
from ui_controls import build_domain_group


class MainWindow(QMainWindow):
    def __init__(self, base_dir: Path):
        super().__init__()
        self.setWindowTitle("Comparador de Derivadas e Convergência de Malha")
        self.resize(1500, 900)
        self.registry = MethodRegistry(base_dir)
        self.canvas = PlotCanvas()
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)
        left, right = QWidget(), QWidget()
        splitter.addWidget(left), splitter.addWidget(right)
        splitter.setSizes([520, 980])
        left_l, right_l = QVBoxLayout(left), QVBoxLayout(right)
        self.domain_group, self.var_controls = build_domain_group()
        left_l.addWidget(self._function_group())
        left_l.addWidget(self.domain_group)
        left_l.addWidget(self._methods_group())
        left_l.addStretch()
        right_l.addWidget(self._controls_group())
        right_l.addWidget(self.canvas, stretch=5)
        right_l.addWidget(self._table_group(), stretch=3)
        self.refresh_method_list()

    def _function_group(self):
        group = QGroupBox("Funções")
        layout = QVBoxLayout(group)
        self.function_input = QLineEdit("sin(x) + y**2 + z*t")
        self.reference_input = QLineEdit("cos(x)")
        help_box = QPlainTextEdit("Função base f(x,y,z,t) e gabarito da derivada.\nUse numpy: sin, cos, exp, pi, ...")
        help_box.setReadOnly(True)
        help_box.setMaximumHeight(90)
        layout.addWidget(QLabel("Função base:"))
        layout.addWidget(self.function_input)
        layout.addWidget(QLabel("Gabarito da derivada (opcional):"))
        layout.addWidget(self.reference_input)
        layout.addWidget(help_box)
        return group

    def _methods_group(self):
        group = QGroupBox("Métodos")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.import_btn = QPushButton("Importar método")
        self.folder_btn = QPushButton("Abrir pasta metodos")
        row.addWidget(self.import_btn), row.addWidget(self.folder_btn)
        self.methods_list = QListWidget()
        layout.addLayout(row), layout.addWidget(self.methods_list)
        self.import_btn.clicked.connect(self.import_method)
        self.folder_btn.clicked.connect(self.open_methods_folder)
        return group

    def _controls_group(self):
        group = QGroupBox("Comparação e refinamento")
        layout = QGridLayout(group)
        self.diff_var = QComboBox(); self.diff_var.addItems(VARIABLES)
        self.plot_var = QComboBox(); self.plot_var.addItems(VARIABLES)
        self.n_values = QLineEdit("21, 41, 81")
        self.positions_input = QLineEdit("0, 0.5, 1.0")
        self.run_btn = QPushButton("Calcular")
        layout.addWidget(QLabel("Derivar em relação a:"), 0, 0); layout.addWidget(self.diff_var, 0, 1)
        layout.addWidget(QLabel("Variável no gráfico:"), 0, 2); layout.addWidget(self.plot_var, 0, 3)
        layout.addWidget(QLabel("Refinamentos N:"), 1, 0); layout.addWidget(self.n_values, 1, 1, 1, 3)
        layout.addWidget(QLabel("Posições/instantes da tabela:"), 2, 0); layout.addWidget(self.positions_input, 2, 1, 1, 3)
        layout.addWidget(self.run_btn, 3, 0, 1, 4)
        self.run_btn.clicked.connect(self.compute)
        return group

    def _table_group(self):
        group = QGroupBox("Tabela de erros")
        layout = QVBoxLayout(group)
        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        return group

    def refresh_method_list(self):
        self.methods_list.clear()
        for method in self.registry.items():
            item = QListWidgetItem(f"{method.name} [{method.source}]")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, method.name)
            self.methods_list.addItem(item)

    def selected_methods(self):
        return [self.registry.methods[self.methods_list.item(i).data(Qt.ItemDataRole.UserRole)] for i in range(self.methods_list.count()) if self.methods_list.item(i).checkState() == Qt.CheckState.Checked]

    def current_configs(self):
        data = {var: control.get_config() for var, control in self.var_controls.items()}
        data[self.plot_var.currentText()]["positions"] = self.positions_input.text()
        return data

    def compute(self):
        try:
            out = run_analysis(self.current_configs(), parse_n_values(self.n_values.text()), self.function_input.text(), self.reference_input.text(), self.diff_var.currentText(), self.plot_var.currentText(), self.selected_methods())
            results_by_n, line_data, conv_data, positions = out
            self.canvas.redraw(line_data, conv_data, self.diff_var.currentText(), self.plot_var.currentText())
            self.update_table(results_by_n, positions)
        except Exception as exc:
            QMessageBox.critical(self, "Erro na execução", str(exc))

    def update_table(self, results_by_n, positions):
        finest_n = max(results_by_n)
        finest_axis = results_by_n[finest_n]["axis_values"]
        coords = [float(finest_axis[idx]) for idx in positions]
        cols = ["N", "coord solicitada", "coord usada", "índice", "gabarito"]
        method_names = list(results_by_n[finest_n]["method_lines"].keys())
        for name in method_names: cols.extend([name, f"erro {name}"])
        rows = len(coords) * len(results_by_n)
        self.table.clear(); self.table.setRowCount(rows); self.table.setColumnCount(len(cols)); self.table.setHorizontalHeaderLabels(cols)
        row = 0
        for n_value in sorted(results_by_n):
            block = results_by_n[n_value]
            axis_values = block["axis_values"]
            for coord in coords:
                idx = int((abs(axis_values - coord)).argmin())
                self.table.setItem(row, 0, QTableWidgetItem(str(n_value)))
                self.table.setItem(row, 1, QTableWidgetItem(f"{coord:.8g}"))
                self.table.setItem(row, 2, QTableWidgetItem(f"{axis_values[idx]:.8g}"))
                self.table.setItem(row, 3, QTableWidgetItem(str(idx)))
                ref = "-" if block["reference"] is None else f"{block['reference'][idx]:.8g}"
                self.table.setItem(row, 4, QTableWidgetItem(ref))
                col = 5
                for name in method_names:
                    val = block["method_lines"][name][idx]
                    err = "-" if name not in block["errors_lines"] else f"{block['errors_lines'][name][idx]:.8g}"
                    self.table.setItem(row, col, QTableWidgetItem(f"{val:.8g}")); self.table.setItem(row, col + 1, QTableWidgetItem(err)); col += 2
                row += 1

    def import_method(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Importar método", "", "Python (*.py)")
        if filepath:
            self.registry.import_file(filepath)
            self.refresh_method_list()

    def open_methods_folder(self):
        folder = str(self.registry.methods_dir)
        try:
            if sys.platform.startswith("win"): os.startfile(folder)
            elif sys.platform == "darwin": __import__("subprocess").Popen(["open", folder])
            else: __import__("subprocess").Popen(["xdg-open", folder])
        except Exception as exc:
            QMessageBox.critical(self, "Erro", f"Não foi possível abrir a pasta: {exc}")
