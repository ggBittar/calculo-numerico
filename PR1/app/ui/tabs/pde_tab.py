from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...backend.pde_methods import PDE_METHODS
from ...backend.pde_service import PdeResult, solve_pde
from ...backend.pdes import BoundaryCondition, PDES, PdeSpec


@dataclass
class BoundaryControls:
    group: QGroupBox
    type_combo: QComboBox
    value_input: QDoubleSpinBox
    alpha_input: QDoubleSpinBox
    beta_input: QDoubleSpinBox
    gamma_input: QDoubleSpinBox


class PdeTab(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._step_inputs: dict[str, QDoubleSpinBox] = {}
        self._count_inputs: dict[str, QSpinBox] = {}
        self._discretization_mode_inputs: dict[str, QComboBox] = {}
        self._variable_tabs: dict[str, QWidget] = {}
        self._boundary_controls: dict[str, BoundaryControls] = {}
        self._pde_ids = list(PDES.keys())
        self._current_result: PdeResult | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(self._build_results(), 1)

        self._refresh_pde_details()
        self._run_solver()

    def _build_controls(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        title = QLabel("Configuração da EDP")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        form_group = QGroupBox("Parâmetros")
        form_layout = QFormLayout(form_group)

        self.pde_combo = QComboBox()
        for pde_id in self._pde_ids:
            self.pde_combo.addItem(PDES[pde_id].name, pde_id)
        form_layout.addRow("EDP", self.pde_combo)

        self.method_combo = QComboBox()
        for method_id, method in PDE_METHODS.items():
            self.method_combo.addItem(method.name, method_id)
        form_layout.addRow("Método numérico", self.method_combo)

        self.equation_label = QLabel()
        self.equation_label.setWordWrap(True)
        form_layout.addRow("Equação", self.equation_label)

        self.boundary_label = QLabel()
        self.boundary_label.setWordWrap(True)
        form_layout.addRow("Contorno padrão", self.boundary_label)

        self.exact_label = QLabel()
        self.exact_label.setWordWrap(True)
        form_layout.addRow("Solução analítica", self.exact_label)

        grid_group = QGroupBox("Malha por variável")
        grid_layout = QVBoxLayout(grid_group)
        self.grid_tabs = QTabWidget()
        self.grid_tabs.setTabPosition(QTabWidget.TabPosition.North)
        grid_layout.addWidget(self.grid_tabs)
        for variable in ("x", "y", "z", "t"):
            variable_tab = QWidget()
            variable_layout = QFormLayout(variable_tab)
            self._variable_tabs[variable] = variable_tab

            mode_combo = QComboBox()
            mode_combo.addItem("Por passo", "step")
            mode_combo.addItem("Por número de passos", "count")
            mode_combo.setEnabled(False)
            self._discretization_mode_inputs[variable] = mode_combo
            variable_layout.addRow("modo", mode_combo)

            step_spin = QDoubleSpinBox()
            step_spin.setRange(0.0001, 10.0)
            step_spin.setDecimals(5)
            step_spin.setSingleStep(0.001)
            step_spin.setValue(0.01)
            step_spin.setEnabled(False)
            self._step_inputs[variable] = step_spin
            variable_layout.addRow(f"d{variable}", step_spin)

            count_spin = QSpinBox()
            count_spin.setRange(1, 10000)
            count_spin.setValue(10)
            count_spin.setEnabled(False)
            self._count_inputs[variable] = count_spin
            variable_layout.addRow("passos", count_spin)
            self.grid_tabs.addTab(variable_tab, variable)

        initial_group = QGroupBox("Condição inicial")
        initial_layout = QVBoxLayout(initial_group)

        self.initial_value_radio = QRadioButton("Valor constante")
        self.initial_function_radio = QRadioButton("Função das variáveis")
        self.initial_value_radio.setChecked(True)

        self.initial_value_input = QDoubleSpinBox()
        self.initial_value_input.setRange(-1_000_000.0, 1_000_000.0)
        self.initial_value_input.setDecimals(6)
        self.initial_value_input.setValue(293.0)

        self.initial_expression_input = QLineEdit("sin(pi*x)")
        self.initial_expression_input.setEnabled(False)

        initial_form = QFormLayout()
        initial_form.addRow(self.initial_value_radio)
        initial_form.addRow("Valor", self.initial_value_input)
        initial_form.addRow(self.initial_function_radio)
        initial_form.addRow("Expressão", self.initial_expression_input)
        initial_layout.addLayout(initial_form)

        self.boundaries_group = QGroupBox("Condições de contorno")
        self.boundaries_layout = QGridLayout(self.boundaries_group)
        self._create_boundary_controls()

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)

        self.run_button = QPushButton("Resolver EDP")

        layout.addWidget(form_group)
        layout.addWidget(grid_group)
        layout.addWidget(initial_group)
        layout.addWidget(self.boundaries_group)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.run_button)
        layout.addStretch(1)

        self.pde_combo.currentIndexChanged.connect(self._refresh_pde_details)
        self.method_combo.currentIndexChanged.connect(self._run_solver)
        self.initial_value_radio.toggled.connect(self._refresh_initial_mode)
        self.initial_function_radio.toggled.connect(self._refresh_initial_mode)
        self.initial_value_input.valueChanged.connect(self._run_solver)
        self.initial_expression_input.textChanged.connect(self._run_solver)
        self.run_button.clicked.connect(self._run_solver)
        for combo in self._discretization_mode_inputs.values():
            combo.currentIndexChanged.connect(self._refresh_discretization_mode)
        for spin in self._step_inputs.values():
            spin.valueChanged.connect(self._run_solver)
        for spin in self._count_inputs.values():
            spin.valueChanged.connect(self._run_solver)

        return container

    def _create_boundary_controls(self) -> None:
        side_labels = {
            "x_min": "x mínimo",
            "x_max": "x máximo",
            "y_min": "y mínimo",
            "y_max": "y máximo",
            "z_min": "z mínimo",
            "z_max": "z máximo",
        }

        for index, side in enumerate(("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
            group = QGroupBox(side_labels[side])
            form = QFormLayout(group)

            type_combo = QComboBox()
            type_combo.addItem("Dirichlet", "dirichlet")
            type_combo.addItem("Neumann", "neumann")
            type_combo.addItem("Robin", "robin")

            value_input = QDoubleSpinBox()
            value_input.setRange(-1_000_000.0, 1_000_000.0)
            value_input.setDecimals(6)

            alpha_input = QDoubleSpinBox()
            alpha_input.setRange(-1_000_000.0, 1_000_000.0)
            alpha_input.setDecimals(6)
            alpha_input.setValue(1.0)

            beta_input = QDoubleSpinBox()
            beta_input.setRange(-1_000_000.0, 1_000_000.0)
            beta_input.setDecimals(6)
            beta_input.setValue(1.0)

            gamma_input = QDoubleSpinBox()
            gamma_input.setRange(-1_000_000.0, 1_000_000.0)
            gamma_input.setDecimals(6)
            gamma_input.setValue(0.0)

            form.addRow("Tipo", type_combo)
            form.addRow("Valor / fluxo", value_input)
            form.addRow("Robin a", alpha_input)
            form.addRow("Robin b", beta_input)
            form.addRow("Robin g", gamma_input)

            controls = BoundaryControls(group, type_combo, value_input, alpha_input, beta_input, gamma_input)
            self._boundary_controls[side] = controls

            self.boundaries_layout.addWidget(group, index // 2, index % 2)

            type_combo.currentIndexChanged.connect(self._refresh_boundary_inputs)
            value_input.valueChanged.connect(self._run_solver)
            alpha_input.valueChanged.connect(self._run_solver)
            beta_input.valueChanged.connect(self._run_solver)
            gamma_input.valueChanged.connect(self._run_solver)

    def _build_results(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMaximumHeight(180)

        self.time_label = QLabel("Tempo selecionado: -")
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self._render_current_time_view)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Índice", "Coordenada", "Aproximação", "Exata/Erro"])

        layout.addWidget(self.summary_box)
        layout.addWidget(self.time_label)
        layout.addWidget(self.time_slider)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.table)
        return container

    def _current_pde(self) -> PdeSpec:
        pde_id = cast(str, self.pde_combo.currentData())
        return PDES[pde_id]

    def _refresh_pde_details(self) -> None:
        spec = self._current_pde()
        self.equation_label.setText(spec.equation)
        self.boundary_label.setText(self._format_boundary_defaults(spec.default_boundary_conditions))
        self.exact_label.setText(spec.exact_expression or "Opcional / não cadastrada")

        for variable, spin in self._step_inputs.items():
            enabled = variable in spec.variables
            spin.setEnabled(enabled)
            if enabled:
                spin.setValue(spec.default_steps[variable])

        for variable, spin in self._count_inputs.items():
            enabled = variable in spec.variables
            spin.setEnabled(enabled)
            if enabled:
                spin.setValue(spec.default_counts[variable])

        for variable, combo in self._discretization_mode_inputs.items():
            enabled = variable in spec.variables
            combo.setEnabled(enabled)
            if enabled:
                combo.setCurrentIndex(combo.findData("step"))
            tab_index = self.grid_tabs.indexOf(self._variable_tabs[variable])
            self.grid_tabs.setTabVisible(tab_index, enabled)

        if spec.variables:
            first_variable = spec.variables[0]
            first_index = self.grid_tabs.indexOf(self._variable_tabs[first_variable])
            self.grid_tabs.setCurrentIndex(first_index)

        for side, controls in self._boundary_controls.items():
            visible = side in spec.boundary_sides
            controls.group.setVisible(visible)
            if visible:
                condition = spec.default_boundary_conditions[side]
                controls.type_combo.setCurrentIndex(controls.type_combo.findData(condition.condition_type))
                controls.value_input.setValue(condition.value)
                controls.alpha_input.setValue(condition.robin_alpha)
                controls.beta_input.setValue(condition.robin_beta)
                controls.gamma_input.setValue(condition.robin_gamma)

        spatial = [name for name in spec.variables if name != "t"]
        example_expression = self._build_example_expression(spatial)
        self.hint_label.setText(
            "Condição inicial pode ser um valor constante ou uma expressão usando: "
            f"{', '.join(spatial)}. Exemplo: {example_expression}"
            if spatial
            else "Sem variáveis espaciais configuradas."
        )
        self.initial_expression_input.setText(example_expression)
        self.initial_value_input.setValue(293.0)
        self._refresh_boundary_inputs()
        self._refresh_discretization_mode()
        self._refresh_initial_mode()

    def _format_boundary_defaults(self, boundaries: dict[str, BoundaryCondition]) -> str:
        return ", ".join(f"{side}: {condition.condition_type}" for side, condition in boundaries.items())

    def _build_example_expression(self, spatial_variables: list[str]) -> str:
        if not spatial_variables:
            return "0.0"
        return " * ".join(f"sin(pi*{variable})" for variable in spatial_variables)

    def _refresh_initial_mode(self) -> None:
        is_value = self.initial_value_radio.isChecked()
        self.initial_value_input.setEnabled(is_value)
        self.initial_expression_input.setEnabled(not is_value)
        self._run_solver()

    def _refresh_discretization_mode(self) -> None:
        for variable in ("x", "y", "z", "t"):
            mode = cast(str, self._discretization_mode_inputs[variable].currentData())
            step_spin = self._step_inputs[variable]
            count_spin = self._count_inputs[variable]
            step_spin.setVisible(step_spin.isEnabled() and mode == "step")
            count_spin.setVisible(count_spin.isEnabled() and mode == "count")
        self._run_solver()

    def _refresh_boundary_inputs(self) -> None:
        spec = self._current_pde()
        for side, controls in self._boundary_controls.items():
            controls.group.setVisible(side in spec.boundary_sides)
            condition_type = cast(str, controls.type_combo.currentData())
            controls.value_input.setVisible(condition_type in {"dirichlet", "neumann"})
            controls.alpha_input.setVisible(condition_type == "robin")
            controls.beta_input.setVisible(condition_type == "robin")
            controls.gamma_input.setVisible(condition_type == "robin")
        self._run_solver()

    def _collect_steps(self) -> dict[str, float]:
        return {variable: spin.value() for variable, spin in self._step_inputs.items() if spin.isEnabled()}

    def _collect_counts(self) -> dict[str, int]:
        return {variable: spin.value() for variable, spin in self._count_inputs.items() if spin.isEnabled()}

    def _collect_discretization_modes(self) -> dict[str, str]:
        return {
            variable: cast(str, combo.currentData())
            for variable, combo in self._discretization_mode_inputs.items()
            if combo.isEnabled()
        }

    def _collect_boundaries(self) -> dict[str, BoundaryCondition]:
        spec = self._current_pde()
        boundaries: dict[str, BoundaryCondition] = {}
        for side in spec.boundary_sides:
            controls = self._boundary_controls[side]
            boundaries[side] = BoundaryCondition(
                condition_type=cast(str, controls.type_combo.currentData()),
                value=controls.value_input.value(),
                robin_alpha=controls.alpha_input.value(),
                robin_beta=controls.beta_input.value(),
                robin_gamma=controls.gamma_input.value(),
            )
        return boundaries

    def _run_solver(self) -> None:
        try:
            result = solve_pde(
                pde_id=cast(str, self.pde_combo.currentData()),
                method_id=cast(str, self.method_combo.currentData()),
                discretization_mode=None,
                steps=self._collect_steps(),
                counts=self._collect_counts(),
                initial_mode="value" if self.initial_value_radio.isChecked() else "function",
                initial_value=self.initial_value_input.value(),
                initial_expression=self.initial_expression_input.text(),
                boundaries=self._collect_boundaries(),
                discretization_modes=self._collect_discretization_modes(),
            )
        except Exception as exc:
            self._current_result = None
            self.summary_box.setPlainText(f"Erro na configuração da EDP:\n{exc}")
            self.figure.clear()
            self.canvas.draw_idle()
            self.table.setRowCount(0)
            self.time_slider.setEnabled(False)
            self.time_label.setText("Tempo selecionado: -")
            return

        self._current_result = result
        self._configure_time_slider()
        self._render_current_time_view()

    def _configure_time_slider(self) -> None:
        if self._current_result is None:
            return

        max_index = len(self._current_result.axes["t"]) - 1
        self.time_slider.blockSignals(True)
        self.time_slider.setEnabled(True)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(max_index)
        self.time_slider.setValue(max_index)
        self.time_slider.blockSignals(False)

    def _render_current_time_view(self) -> None:
        if self._current_result is None:
            return

        time_index = self.time_slider.value()
        self._render_summary(self._current_result, time_index)
        self._render_plot(self._current_result, time_index)
        self._render_table(self._current_result, time_index)

    def _render_summary(self, result: PdeResult, time_index: int) -> None:
        steps_text = ", ".join(
            f"d{axis}={result.axes[axis][1] - result.axes[axis][0]:.5f}" for axis in result.pde_spec.variables
        )
        counts_text = ", ".join(f"{axis}:{len(result.axes[axis]) - 1}" for axis in result.pde_spec.variables)
        error_text = f"{result.error_max:.3e}" if result.error_max is not None else "Nao disponivel"
        selected_time = result.axes["t"][time_index]
        self.time_label.setText(
            f"Tempo selecionado: t={selected_time:.5f} ({time_index}/{len(result.axes['t']) - 1})"
        )
        summary = (
            f"EDP: {result.metadata['equation']}\n"
            f"Metodo: {result.method_spec.name}\n"
            f"Passos: {steps_text}\n"
            f"Número de passos: {counts_text}\n"
            f"Condicao inicial: {result.metadata['initial_mode']}\n"
            f"Contornos: {result.metadata['boundary']}\n"
            f"Solução analítica: {result.metadata['exact_status']}\n"
            f"Erro maximo final: {error_text}"
        )
        self.summary_box.setPlainText(summary)

    def _render_plot(self, result: PdeResult, time_index: int) -> None:
        self.figure.clear()
        self.figure.patch.set_facecolor("#121417")
        selected_time = result.axes["t"][time_index]
        selected_slice = result.solution[time_index]
        selected_exact_slice = result.exact_solution[time_index] if result.exact_solution is not None else None

        if len(result.spatial_axes) == 1:
            axis_name = result.spatial_axes[0]
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#1b1f24")
            axis_values = result.axes[axis_name]
            ax.plot(axis_values, selected_slice, label="Aproximação", color="#88c0d0", linewidth=2.0)
            if selected_exact_slice is not None:
                ax.plot(axis_values, selected_exact_slice, label="Exata", color="#a3be8c", linestyle="--")
            ax.set_title(f"Perfil em t={selected_time:.4f}")
            ax.set_xlabel(axis_name)
            ax.set_ylabel("u")
            ax.tick_params(colors="#e5e9f0")
            ax.xaxis.label.set_color("#e5e9f0")
            ax.yaxis.label.set_color("#e5e9f0")
            ax.title.set_color("#eceff4")
            for spine in ax.spines.values():
                spine.set_color("#4c566a")
            ax.grid(True, alpha=0.25, color="#4c566a")
            ax.legend()
            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor("#2e3440")
                legend.get_frame().set_edgecolor("#4c566a")
                for text in legend.get_texts():
                    text.set_color("#eceff4")
        elif len(result.spatial_axes) == 2:
            axis_x, axis_y = result.spatial_axes
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#1b1f24")
            image = ax.imshow(
                selected_slice.T,
                origin="lower",
                aspect="auto",
                extent=[
                    result.axes[axis_x][0],
                    result.axes[axis_x][-1],
                    result.axes[axis_y][0],
                    result.axes[axis_y][-1],
                ],
                cmap="viridis",
            )
            ax.set_title(f"Campo em t={selected_time:.4f}")
            ax.set_xlabel(axis_x)
            ax.set_ylabel(axis_y)
            ax.tick_params(colors="#e5e9f0")
            ax.xaxis.label.set_color("#e5e9f0")
            ax.yaxis.label.set_color("#e5e9f0")
            ax.title.set_color("#eceff4")
            for spine in ax.spines.values():
                spine.set_color("#4c566a")
            colorbar = self.figure.colorbar(image, ax=ax, label="u")
            colorbar.ax.yaxis.label.set_color("#e5e9f0")
            colorbar.ax.tick_params(colors="#e5e9f0")
            colorbar.outline.set_edgecolor("#4c566a")
        else:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#1b1f24")
            ax.tick_params(colors="#e5e9f0")
            ax.text(0.5, 0.5, "Visualização não implementada para esta dimensão.", ha="center", va="center")
            ax.set_axis_off()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_table(self, result: PdeResult, time_index: int) -> None:
        selected_slice = result.solution[time_index]
        selected_exact_slice = result.exact_solution[time_index] if result.exact_solution is not None else None

        if len(result.spatial_axes) == 1:
            axis_name = result.spatial_axes[0]
            axis_values = result.axes[axis_name]
            row_count = min(12, len(axis_values))
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["Índice", axis_name, "Aproximação", "Exata/Erro"])
            self.table.setRowCount(row_count)

            for row in range(row_count):
                exact_or_error = "N/D"
                if selected_exact_slice is not None:
                    exact_value = float(selected_exact_slice[row])
                    approx_value = float(selected_slice[row])
                    exact_or_error = f"{exact_value:.6f} / {abs(approx_value - exact_value):.2e}"

                values = [
                    str(row),
                    f"{axis_values[row]:.6f}",
                    f"{float(selected_slice[row]):.6f}",
                    exact_or_error,
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, col, item)
        else:
            nx, ny = selected_slice.shape
            row_count = min(12, nx * ny)
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(
                ["Índice", result.spatial_axes[0], result.spatial_axes[1], "Aproximação", "Exata/Erro"]
            )
            self.table.setRowCount(row_count)

            flat = selected_slice.reshape(-1)
            exact_flat = selected_exact_slice.reshape(-1) if selected_exact_slice is not None else None
            x_values = np.repeat(result.axes[result.spatial_axes[0]], ny)
            y_values = np.tile(result.axes[result.spatial_axes[1]], nx)

            for row in range(row_count):
                exact_or_error = "N/D"
                if exact_flat is not None:
                    exact_or_error = f"{float(exact_flat[row]):.6f} / {abs(float(flat[row]) - float(exact_flat[row])):.2e}"

                values = [
                    str(row),
                    f"{x_values[row]:.4f}",
                    f"{y_values[row]:.4f}",
                    f"{float(flat[row]):.6f}",
                    exact_or_error,
                ]
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()
