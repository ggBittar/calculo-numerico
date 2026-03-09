from __future__ import annotations

import csv
from pathlib import Path

from matplotlib.figure import Figure


def _table_matrix(table_widget):
    headers = []
    for col in range(table_widget.columnCount()):
        item = table_widget.horizontalHeaderItem(col)
        headers.append("" if item is None else item.text())
    rows = []
    for row in range(table_widget.rowCount()):
        values = []
        for col in range(table_widget.columnCount()):
            item = table_widget.item(row, col)
            values.append("" if item is None else item.text())
        rows.append(values)
    return headers, rows


def export_table_csv(table_widget, filepath: str | Path):
    headers, rows = _table_matrix(table_widget)
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def export_table_png(table_widget, filepath: str | Path, width: int = 1920, height: int = 1080):
    headers, rows = _table_matrix(table_widget)
    if not headers:
        raise ValueError("A tabela está vazia.")

    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#202124")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#202124")
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    font_size = 8 if len(headers) <= 10 else 6
    table.set_fontsize(font_size)
    x_scale = min(1.8, max(1.0, 10 / max(1, len(headers))))
    y_scale = min(1.5, max(0.8, 24 / max(1, len(rows) + 1)))
    table.scale(x_scale, y_scale)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2d2f31")
            cell.get_text().set_color("#e8eaed")
        else:
            cell.set_facecolor("#171717")
            cell.get_text().set_color("#e8eaed")
        cell.set_edgecolor("#5f6368")

    fig.tight_layout()
    fig.savefig(filepath, dpi=100, facecolor=fig.get_facecolor())
