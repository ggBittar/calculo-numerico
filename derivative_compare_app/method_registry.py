from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List
import importlib.util

import numpy as np


@dataclass
class DerivativeMethod:
    name: str
    func: Callable
    source: str


class MethodRegistry:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.methods_dir = self.base_dir / "metodos"
        self.methods_dir.mkdir(exist_ok=True)
        self.methods: Dict[str, DerivativeMethod] = {}
        self._register_builtin()
        self.load_folder()

    def _register_builtin(self):
        def central(_, values, axis, spacings):
            h = spacings[axis]
            result = np.empty_like(values, dtype=float)
            mid = [slice(None)] * values.ndim
            prev_ = [slice(None)] * values.ndim
            next_ = [slice(None)] * values.ndim
            mid[axis], prev_[axis], next_[axis] = slice(1, -1), slice(0, -2), slice(2, None)
            result[tuple(mid)] = (values[tuple(next_)] - values[tuple(prev_)]) / (2 * h)
            left = [slice(None)] * values.ndim
            left_n = [slice(None)] * values.ndim
            left[axis], left_n[axis] = 0, 1
            result[tuple(left)] = (values[tuple(left_n)] - values[tuple(left)]) / h
            right = [slice(None)] * values.ndim
            right_p = [slice(None)] * values.ndim
            right[axis], right_p[axis] = -1, -2
            result[tuple(right)] = (values[tuple(right)] - values[tuple(right_p)]) / h
            return result

        def forward(_, values, axis, spacings):
            h = spacings[axis]
            result = np.empty_like(values, dtype=float)
            cur = [slice(None)] * values.ndim
            nxt = [slice(None)] * values.ndim
            cur[axis], nxt[axis] = slice(0, -1), slice(1, None)
            result[tuple(cur)] = (values[tuple(nxt)] - values[tuple(cur)]) / h
            last = [slice(None)] * values.ndim
            prev_ = [slice(None)] * values.ndim
            last[axis], prev_[axis] = -1, -2
            result[tuple(last)] = (values[tuple(last)] - values[tuple(prev_)]) / h
            return result

        def backward(_, values, axis, spacings):
            h = spacings[axis]
            result = np.empty_like(values, dtype=float)
            cur = [slice(None)] * values.ndim
            prev_ = [slice(None)] * values.ndim
            cur[axis], prev_[axis] = slice(1, None), slice(0, -1)
            result[tuple(cur)] = (values[tuple(cur)] - values[tuple(prev_)]) / h
            first = [slice(None)] * values.ndim
            next_ = [slice(None)] * values.ndim
            first[axis], next_[axis] = 0, 1
            result[tuple(first)] = (values[tuple(next_)] - values[tuple(first)]) / h
            return result

        self.add_method("Diferença Central", central, "interno")
        self.add_method("Diferença Progressiva", forward, "interno")
        self.add_method("Diferença Regressiva", backward, "interno")

    def add_method(self, name: str, func: Callable, source: str):
        self.methods[name] = DerivativeMethod(name=name, func=func, source=source)

    def _load_one(self, file_path: Path):
        module_name = f"method_{file_path.stem}_{abs(hash(str(file_path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Falha ao carregar {file_path.name}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "estimate"):
            raise RuntimeError(f"{file_path.name} precisa definir estimate(...).")
        name = getattr(module, "METHOD_NAME", file_path.stem)
        self.add_method(name, module.estimate, str(file_path))

    def load_folder(self):
        for file_path in sorted(self.methods_dir.glob("*.py")):
            try:
                self._load_one(file_path)
            except Exception as exc:
                print(f"Aviso ao carregar {file_path.name}: {exc}")

    def import_file(self, filepath: str):
        src = Path(filepath)
        target = self.methods_dir / src.name
        if src.resolve() != target.resolve():
            target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        self._load_one(target)

    def items(self) -> List[DerivativeMethod]:
        return list(self.methods.values())
