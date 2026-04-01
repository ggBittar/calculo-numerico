from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .models import BoundaryConfig, SimulationResult, ThermalConfig
from .solver import left_ghost_temperature, profile_to_state, right_ghost_temperature, solve_problem


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("aula_5 - Conducao 1D com Celulas Fantasmas")
        self.resize(1440, 900)

        self._current_result: SimulationResult | None = None
        self.left_boundary_inputs: dict[str, QLineEdit] = {}
        self.right_boundary_inputs: dict[str, QLineEdit] = {}

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(self._build_results(), 1)

        self.setCentralWidget(central)
        self._run_simulation()

    def _build_controls(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        title = QLabel("Parametros do problema")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        boundary_group = QGroupBox("Condicoes de contorno (esquerda e direita)")
        boundary_grid = QGridLayout(boundary_group)
        boundary_grid.setHorizontalSpacing(12)

        left_group = QGroupBox("Contorno esquerdo")
        left_form = QFormLayout(left_group)
        self.left_boundary_inputs = self._build_boundary_inputs(
            left_form,
            defaults={
                "emissividade": 0,
                "sigma": 5.67e-8,
                "h": 0.0,
                "fluxo_superficial": 0.0,
                "contato": 1.0,
                "temperatura_vizinhanca": 0.0,
                "temperatura_infinito": 300.0,
                "temperatura_contorno": 273.0,
            },
        )

        right_group = QGroupBox("Contorno direito")
        right_form = QFormLayout(right_group)
        self.right_boundary_inputs = self._build_boundary_inputs(
            right_form,
            defaults={
                "emissividade": 0.0,
                "sigma": 5.67e-8,
                "h": 0.0,
                "fluxo_superficial": 0.0,
                "contato": 1.0,
                "temperatura_vizinhanca": 0.0,
                "temperatura_infinito": 300.0,
                "temperatura_contorno": 373.0,
            },
        )

        boundary_grid.addWidget(left_group, 0, 0)
        boundary_grid.addWidget(right_group, 0, 1)

        global_group = QGroupBox("Parametros globais")
        global_form = QFormLayout(global_group)
        self.area_input = self._make_text_input(1e-3)
        self.k_input = self._make_text_input(1e3)
        self.t0_input = self._make_text_input(300.0)
        global_form.addRow("Area A", self.area_input)
        global_form.addRow("Condutividade k", self.k_input)
        global_form.addRow("T inicial", self.t0_input)

        discretization_group = QGroupBox("Discretizacao")
        discretization_form = QFormLayout(discretization_group)
        self.to_input = self._make_text_input(0.0)
        self.tf_input = self._make_text_input(1.0)
        self.co_input = self._make_text_input(0.25)
        self.xo_input = self._make_text_input(0.0)
        self.xf_input = self._make_text_input(1.0)
        self.nx_input = self._make_text_input(100)
        discretization_form.addRow("t0", self.to_input)
        discretization_form.addRow("tf", self.tf_input)
        discretization_form.addRow("Co", self.co_input)
        discretization_form.addRow("x0", self.xo_input)
        discretization_form.addRow("xf", self.xf_input)
        discretization_form.addRow("nx", self.nx_input)

        self.run_button = QPushButton("Calcular e plotar")
        self.run_button.clicked.connect(self._run_simulation)

        for widget in self._all_input_widgets():
            widget.editingFinished.connect(self._run_simulation)

        layout.addWidget(boundary_group)
        layout.addWidget(global_group)
        layout.addWidget(discretization_group)
        layout.addWidget(self.run_button)
        layout.addStretch(1)
        return container

    def _build_boundary_inputs(
        self,
        form: QFormLayout,
        defaults: dict[str, float],
    ) -> dict[str, QLineEdit]:
        inputs = {
            "emissividade": self._make_text_input(defaults["emissividade"]),
            "sigma": self._make_text_input(defaults["sigma"]),
            "h": self._make_text_input(defaults["h"]),
            "fluxo_superficial": self._make_text_input(defaults["fluxo_superficial"]),
            "contato": self._make_text_input(defaults["contato"]),
            "temperatura_vizinhanca": self._make_text_input(defaults["temperatura_vizinhanca"]),
            "temperatura_infinito": self._make_text_input(defaults["temperatura_infinito"]),
            "temperatura_contorno": self._make_text_input(defaults["temperatura_contorno"]),
        }
        form.addRow("Emissividade e", inputs["emissividade"])
        form.addRow("Sigma", inputs["sigma"])
        form.addRow("h", inputs["h"])
        form.addRow("qs", inputs["fluxo_superficial"])
        form.addRow("Contato C", inputs["contato"])
        form.addRow("Tviz", inputs["temperatura_vizinhanca"])
        form.addRow("Tinf", inputs["temperatura_infinito"])
        form.addRow("T ref.", inputs["temperatura_contorno"])
        return inputs

    def _all_input_widgets(self) -> list[QLineEdit]:
        return [
            *self.left_boundary_inputs.values(),
            *self.right_boundary_inputs.values(),
            self.area_input,
            self.k_input,
            self.t0_input,
            self.to_input,
            self.tf_input,
            self.co_input,
            self.xo_input,
            self.xf_input,
            self.nx_input,
        ]

    def _build_results(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMaximumHeight(280)

        self.time_label = QLabel("Tempo selecionado: -")
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self._render_current_time)

        self.figure = Figure(figsize=(8.5, 5.2))
        self.canvas = FigureCanvas(self.figure)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Indice", "x", "T(x,t)"])

        layout.addWidget(self.summary_box)
        layout.addWidget(self.time_label)
        layout.addWidget(self.time_slider)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.table)
        return container

    def _make_text_input(self, value: float | int) -> QLineEdit:
        text = f"{float(value):.15g}" if isinstance(value, float) else str(value)
        line = QLineEdit(text)
        line.setMinimumWidth(120)
        return line

    def _parse_float(self, input_widget: QLineEdit, field_name: str) -> float:
        raw = input_widget.text().strip()
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(
                f"Valor invalido em '{field_name}': '{raw}'. Use numero real (ex.: 5.67e-8)."
            ) from exc

    def _parse_int(self, input_widget: QLineEdit, field_name: str) -> int:
        raw = input_widget.text().strip()
        try:
            value_float = float(raw)
        except ValueError as exc:
            raise ValueError(f"Valor invalido em '{field_name}': '{raw}'. Use numero inteiro.") from exc
        value_int = int(value_float)
        if not np.isclose(value_float, value_int):
            raise ValueError(f"Valor invalido em '{field_name}': '{raw}'. Use numero inteiro.")
        return value_int

    def _collect_boundary_config(self, inputs: dict[str, QLineEdit], side_label: str) -> BoundaryConfig:
        return BoundaryConfig(
            emissividade=self._parse_float(inputs["emissividade"], f"emissividade ({side_label})"),
            sigma=self._parse_float(inputs["sigma"], f"sigma ({side_label})"),
            h=self._parse_float(inputs["h"], f"h ({side_label})"),
            fluxo_superficial=self._parse_float(inputs["fluxo_superficial"], f"qs ({side_label})"),
            contato=self._parse_float(inputs["contato"], f"contato C ({side_label})"),
            temperatura_vizinhanca=self._parse_float(inputs["temperatura_vizinhanca"], f"Tviz ({side_label})"),
            temperatura_infinito=self._parse_float(inputs["temperatura_infinito"], f"Tinf ({side_label})"),
            temperatura_contorno=self._parse_float(inputs["temperatura_contorno"], f"T ref. ({side_label})"),
        )

    def _collect_config(self) -> ThermalConfig:
        nx = self._parse_int(self.nx_input, "nx")
        return ThermalConfig(
            area=self._parse_float(self.area_input, "Area A"),
            condutividade=self._parse_float(self.k_input, "Condutividade k"),
            temperatura_inicial=self._parse_float(self.t0_input, "T inicial"),
            tempo_inicial=self._parse_float(self.to_input, "t0"),
            tempo_final=self._parse_float(self.tf_input, "tf"),
            co=self._parse_float(self.co_input, "Co"),
            x_inicial=self._parse_float(self.xo_input, "x0"),
            x_final=self._parse_float(self.xf_input, "xf"),
            nx=nx,
            esquerda=self._collect_boundary_config(self.left_boundary_inputs, "esquerda"),
            direita=self._collect_boundary_config(self.right_boundary_inputs, "direita"),
        )

    def _run_simulation(self) -> None:
        try:
            result = solve_problem(self._collect_config())
        except Exception as exc:
            self._current_result = None
            self.summary_box.setPlainText(f"Erro na configuracao:\n{exc}")
            self.figure.clear()
            self.canvas.draw_idle()
            self.table.setRowCount(0)
            self.time_slider.setEnabled(False)
            self.time_label.setText("Tempo selecionado: -")
            return

        self._current_result = result
        self._configure_time_slider()
        self._render_current_time()

    def _configure_time_slider(self) -> None:
        if self._current_result is None:
            return

        max_index = len(self._current_result.t_axis) - 1
        self.time_slider.blockSignals(True)
        self.time_slider.setEnabled(True)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(max_index)
        self.time_slider.setValue(max_index)
        self.time_slider.blockSignals(False)

    def _render_current_time(self) -> None:
        if self._current_result is None:
            return

        time_index = self.time_slider.value()
        self._render_summary(self._current_result, time_index)
        self._render_plot(self._current_result, time_index)
        self._render_table(self._current_result, time_index)

    def _render_summary(self, result: SimulationResult, time_index: int) -> None:
        selected_time = float(result.t_axis[time_index])
        selected_profile = result.solution[time_index]
        sampled_state = profile_to_state(selected_profile)
        left_ghost = left_ghost_temperature(sampled_state, result.config, result.dx)
        right_ghost = right_ghost_temperature(sampled_state, result.config, result.dx)

        self.time_label.setText(
            f"Tempo selecionado: t={selected_time:.5f} ({time_index}/{len(result.t_axis) - 1})"
        )
        summary = (
            "Problema: conducao transiente 1D com celulas fantasmas\n"
            f"dx = {result.dx:.6f}\n"
            f"dt calculado = Co*dx^2/(k*A) = {result.dt:.6g}\n"
            f"nt = {len(result.t_axis)}\n"
            f"Co informado = {result.config.co:.6f}\n"
            f"fator efetivo = k*A*dt/dx^2 = {result.factor:.6f}\n"
            f"Formula ghost esquerda: {result.left_ghost_formula}\n"
            f"Formula ghost direita: {result.right_ghost_formula}\n"
            f"Ghost esquerdo no tempo atual = {left_ghost:.6f}\n"
            f"Ghost direito no tempo atual = {right_ghost:.6f}\n"
            f"Tmin(t) = {float(np.min(selected_profile)):.6f}\n"
            f"Tmax(t) = {float(np.max(selected_profile)):.6f}\n"
            f"T media(t) = {float(np.mean(selected_profile)):.6f}"
        )
        self.summary_box.setPlainText(summary)

    def _render_plot(self, result: SimulationResult, time_index: int) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1b1f24")
        self.figure.patch.set_facecolor("#121417")

        selected_time = float(result.t_axis[time_index])
        selected_profile = result.solution[time_index]

        ax.plot(result.x_axis, selected_profile, color="#88c0d0", linewidth=2.2, label="T(x,t)")
        ax.scatter(result.x_axis, selected_profile, color="#d08770", s=8, zorder=3)
        ax.set_title(f"Perfil de temperatura em t = {selected_time:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("Temperatura")
        ax.set_ylim(*result.plot_y_limits)
        ax.grid(True, alpha=0.25, color="#4c566a")
        ax.tick_params(colors="#e5e9f0")
        ax.xaxis.label.set_color("#e5e9f0")
        ax.yaxis.label.set_color("#e5e9f0")
        ax.title.set_color("#eceff4")
        for spine in ax.spines.values():
            spine.set_color("#4c566a")
        legend = ax.legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#2e3440")
            legend.get_frame().set_edgecolor("#4c566a")
            for text in legend.get_texts():
                text.set_color("#eceff4")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_table(self, result: SimulationResult, time_index: int) -> None:
        selected_profile = result.solution[time_index]
        row_count = len(result.x_axis)
        self.table.setRowCount(row_count)

        for row, (x_value, temperature) in enumerate(zip(result.x_axis, selected_profile, strict=True)):
            values = [str(row), f"{float(x_value):.6f}", f"{float(temperature):.6f}"]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)

        self.table.resizeColumnsToContents()
