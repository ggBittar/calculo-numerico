from __future__ import annotations

from typing import Dict

from PyQt6.QtWidgets import QCheckBox, QDoubleSpinBox, QGridLayout, QLabel, QGroupBox, QVBoxLayout, QWidget

from expression_utils import VARIABLES


class VariableControl(QWidget):
    def __init__(self, var_name: str):
        super().__init__()
        self.var_name = var_name
        layout = QGridLayout(self)
        self.enabled = QCheckBox(f"Habilitar {var_name}")
        self.enabled.setChecked(var_name == "x")
        self.min_spin = self._make_spin(0.0)
        self.max_spin = self._make_spin(1.0)
        self.slice_ratio = self._make_spin(0.5, 0.0, 1.0)
        layout.addWidget(self.enabled, 0, 0, 1, 2)
        layout.addWidget(QLabel("Mín:"), 1, 0)
        layout.addWidget(self.min_spin, 1, 1)
        layout.addWidget(QLabel("Máx:"), 2, 0)
        layout.addWidget(self.max_spin, 2, 1)
        layout.addWidget(QLabel("Posição do corte (0 a 1):"), 3, 0)
        layout.addWidget(self.slice_ratio, 3, 1)
        self.enabled.stateChanged.connect(self._sync_enabled)
        self._sync_enabled()

    def _make_spin(self, value: float, low: float = -1e6, high: float = 1e6):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(6)
        spin.setValue(value)
        spin.setSingleStep(0.1)
        return spin

    def _sync_enabled(self):
        active = self.enabled.isChecked()
        for w in [self.min_spin, self.max_spin, self.slice_ratio]:
            w.setEnabled(active)

    def get_config(self) -> dict:
        return {
            "enabled": self.enabled.isChecked(),
            "min": self.min_spin.value(),
            "max": self.max_spin.value(),
            "slice_ratio": self.slice_ratio.value(),
        }


def build_domain_group(parent=None):
    group = QGroupBox("Domínio")
    layout = QGridLayout(group)
    controls: Dict[str, VariableControl] = {}
    for idx, var in enumerate(VARIABLES):
        control = VariableControl(var)
        controls[var] = control
        layout.addWidget(control, idx // 2, idx % 2)
    return group, controls


def apply_dark_theme(app):
    app.setStyleSheet(
        """
        QWidget { background: #202124; color: #e8eaed; font-size: 12px; }
        QGroupBox { border: 1px solid #3c4043; margin-top: 10px; padding-top: 12px; border-radius: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QPushButton { background: #3c4043; border: 1px solid #5f6368; padding: 8px; border-radius: 6px; }
        QPushButton:hover { background: #4b4f54; }
        QLineEdit, QListWidget, QTableWidget, QComboBox, QPlainTextEdit, QDoubleSpinBox { background: #171717; border: 1px solid #5f6368; border-radius: 6px; padding: 4px; }
        QHeaderView::section { background: #2d2f31; color: #e8eaed; padding: 6px; border: 0; }
        QCheckBox::indicator { width: 16px; height: 16px; }
        """
    )
