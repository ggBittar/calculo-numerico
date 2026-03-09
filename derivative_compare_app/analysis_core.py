from __future__ import annotations

from typing import Dict, List

import numpy as np

from expression_utils import build_grid, evaluate_expression, expand_scalar, extract_line, parse_positions


def _build_block(methods, axes_vectors, values, diff_axis, spacings, axis_order, plot_var, configs, n_value, reference):
    axis_values = axes_vectors[plot_var]
    methods_lines, errors_lines = {}, {}
    ref_line = None if reference is None else extract_line(reference, axis_order, plot_var, configs, n_value)
    line_data, conv_pairs = [], []

    for method in methods:
        deriv = np.asarray(method.func(axes_vectors, values, diff_axis, spacings), dtype=float)
        if deriv.shape != values.shape:
            raise ValueError(f"{method.name} retornou shape inválido: {deriv.shape}.")
        method_line = extract_line(deriv, axis_order, plot_var, configs, n_value)
        methods_lines[method.name] = method_line
        line_data.append((f"{method.name} | N={n_value}", axis_values, method_line))
        if ref_line is not None:
            err_line = np.abs(method_line - ref_line)
            errors_lines[method.name] = err_line
            conv_pairs.append((method.name, n_value, float(np.max(err_line))))

    if ref_line is not None:
        line_data.append((f"Gabarito | N={n_value}", axis_values, ref_line))
    return {
        "axis_values": axis_values,
        "reference": ref_line,
        "method_lines": methods_lines,
        "errors_lines": errors_lines,
    }, line_data, conv_pairs


def run_analysis(configs: Dict[str, dict], n_values: List[int], function_expr: str, reference_expr: str,
                 diff_var: str, plot_var: str, methods: List):
    if not methods:
        raise ValueError("Selecione ao menos um método.")
    if not function_expr.strip():
        raise ValueError("Informe a função base.")

    results_by_n, line_data = {}, []
    conv_series = {m.name: [[], []] for m in methods}

    for n_value in n_values:
        axis_order, axes_vectors, spacings, locals_map = build_grid(configs, n_value)
        if diff_var not in axis_order or plot_var not in axis_order:
            raise ValueError("A variável de derivação e a de gráfico precisam estar habilitadas.")

        values = expand_scalar(evaluate_expression(function_expr, locals_map), axis_order, axes_vectors)
        diff_axis = axis_order.index(diff_var)
        reference = None
        if reference_expr.strip():
            reference = expand_scalar(evaluate_expression(reference_expr, locals_map), axis_order, axes_vectors)

        block, lines, conv_pairs = _build_block(
            methods, axes_vectors, values, diff_axis, spacings, axis_order, plot_var, configs, n_value, reference
        )
        results_by_n[n_value] = block
        line_data.extend(lines)
        for name, n_item, err in conv_pairs:
            conv_series[name][0].append(n_item)
            conv_series[name][1].append(err)

    finest_n = max(n_values)
    finest_axis = results_by_n[finest_n]["axis_values"]
    positions = parse_positions(configs[plot_var]["positions"], finest_axis)
    return results_by_n, line_data, [(k, *v) for k, v in conv_series.items()], positions
