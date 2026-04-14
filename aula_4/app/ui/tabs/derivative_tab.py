from __future__ import annotations

from typing import cast

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...backend.derivative_methods import METHODS
from ...backend.derivative_service import DerivativeResult, estimate_derivative
from ...backend.functions import FUNCTIONS, FunctionSpec


class DerivativeTab(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._ni_inputs: dict[str, QSpinBox] = {}
        self._function_ids: list[str] = list(FUNCTIONS.keys())
        self._method_ids: list[str] = list(METHODS.keys())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        controls = self._build_controls()
        results = self._build_results()

        layout.addWidget(controls, 0)
        layout.addWidget(results, 1)

        self._refresh_function_details()
        self._run_estimation()

    def _build_controls(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        title = QLabel("Configuração da estimativa")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        form_group = QGroupBox("Parâmetros")
        form_layout = QFormLayout(form_group)
        form_layout.setSpacing(10)

        self.function_combo = QComboBox()
        for function_id in self._function_ids:
            self.function_combo.addItem(FUNCTIONS[function_id].name, function_id)
        form_layout.addRow("Escolha da função", self.function_combo)

        self.method_combo = QComboBox()
        for method_id in self._method_ids:
            self.method_combo.addItem(METHODS[method_id].name, method_id)
        form_layout.addRow("Método de estimativa", self.method_combo)

        ni_group = QGroupBox("Ni por dimensão")
        ni_layout = QFormLayout(ni_group)
        ni_layout.setSpacing(8)
        for variable in ("x", "y", "z", "t"):
            spin = QSpinBox()
            spin.setRange(3, 2001)
            spin.setSingleStep(2)
            spin.setValue(21)
            spin.setEnabled(False)
            self._ni_inputs[variable] = spin
            ni_layout.addRow(f"N{variable}", spin)

        self.expression_label = QLabel()
        self.expression_label.setWordWrap(True)
        form_layout.addRow("Função", self.expression_label)

        self.derivative_label = QLabel()
        self.derivative_label.setWordWrap(True)
        form_layout.addRow("Derivada analítica", self.derivative_label)

        self.axis_label = QLabel()
        form_layout.addRow("Derivar em", self.axis_label)

        self.slice_label = QLabel()
        self.slice_label.setWordWrap(True)

        self.run_button = QPushButton("Executar estimativa")

        layout.addWidget(form_group)
        layout.addWidget(ni_group)
        layout.addWidget(self.slice_label)
        layout.addWidget(self.run_button)
        layout.addStretch(1)

        self.function_combo.currentIndexChanged.connect(self._refresh_function_details)
        self.method_combo.currentIndexChanged.connect(self._run_estimation)
        self.run_button.clicked.connect(self._run_estimation)
        for spin in self._ni_inputs.values():
            spin.valueChanged.connect(self._run_estimation)

        return container

    def _build_results(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMaximumHeight(150)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            [
                "Ponto",
                "Eixo",
                "f",
                "Aprox.",
                "Exata/Erro",
            ]
        )

        layout.addWidget(self.summary_box)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.table)

        return container

    def _current_function(self) -> FunctionSpec:
        function_id = cast(str, self.function_combo.currentData())
        return FUNCTIONS[function_id]

    def _collect_ni(self) -> dict[str, int]:
        return {variable: spin.value() for variable, spin in self._ni_inputs.items()}

    def _refresh_function_details(self) -> None:
        function_spec = self._current_function()

        self.expression_label.setText(function_spec.expression)
        self.derivative_label.setText(function_spec.derivative_expression or "Opcional / não cadastrada")
        self.axis_label.setText(function_spec.derivative_variable)

        for variable, spin in self._ni_inputs.items():
            enabled = variable in function_spec.variables
            spin.setEnabled(enabled)
            if enabled and spin.value() < 3:
                spin.setValue(21)

        fixed_variables = [
            variable
            for variable in function_spec.variables
            if variable != function_spec.derivative_variable
        ]
        if fixed_variables:
            self.slice_label.setText(
                "As outras dimensões são avaliadas no corte central da malha: "
                f"{', '.join(fixed_variables)}."
            )
        else:
            self.slice_label.setText("Função unidimensional: sem cortes adicionais.")

        self._run_estimation()

    def _run_estimation(self) -> None:
        function_id = cast(str, self.function_combo.currentData())
        method_id = cast(str, self.method_combo.currentData())

        result = estimate_derivative(function_id, method_id, self._collect_ni())
        self._render_summary(result)
        self._render_plot(result)
        self._render_table(result)

    def _render_summary(self, result: DerivativeResult) -> None:
        approx_text = result.method_spec.description
        analytical = result.metadata["derivative_expression"]
        summary = (
            f"Funcao: {result.metadata['function_expression']}\n"
            f"Metodo: {result.method_spec.name} ({approx_text})\n"
            f"Derivada alvo: d/d{result.axis_name}\n"
            f"Derivada analitica: {analytical}\n"
            f"Corte utilizado: {result.slice_description}"
        )
        self.summary_box.setPlainText(summary)

    def _render_plot(self, result: DerivativeResult) -> None:
        self.axes.clear()

        self.axes.plot(
            result.axis_values,
            result.sampled_values,
            label="f amostrada",
            color="#4c566a",
            alpha=0.55,
        )
        self.axes.plot(
            result.approx_axis_values,
            result.approx_values,
            label=result.method_spec.name,
            color="#d08770",
            linewidth=2.0,
        )

        if result.exact_values is not None:
            self.axes.plot(
                result.axis_values,
                result.exact_values,
                label="Derivada analítica",
                color="#5e81ac",
                linestyle="--",
                linewidth=2.0,
            )

        self.axes.set_title(
            f"Estimativa em relacao a {result.axis_name} para {result.function_spec.name}"
        )
        self.axes.set_xlabel(result.axis_name)
        self.axes.set_ylabel("Valor")
        self.axes.grid(True, alpha=0.25)
        self.axes.legend()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_table(self, result: DerivativeResult) -> None:
        row_count = min(12, len(result.approx_values))
        self.table.setRowCount(row_count)

        for row in range(row_count):
            axis_value = result.approx_axis_values[row]
            approx_value = result.approx_values[row]
            point_value = np.interp(axis_value, result.axis_values, result.sampled_values)

            exact_or_error = "N/D"
            if result.exact_values is not None:
                exact_value = np.interp(axis_value, result.axis_values, result.exact_values)
                exact_or_error = f"{exact_value:.6f} / {abs(approx_value - exact_value):.2e}"

            items = [
                str(row),
                f"{axis_value:.6f}",
                f"{point_value:.6f}",
                f"{approx_value:.6f}",
                exact_or_error,
            ]

            for column, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)

        self.table.resizeColumnsToContents()
