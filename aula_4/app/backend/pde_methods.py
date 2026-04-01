from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PdeMethodSpec:
    method_id: str
    name: str
    description: str


PDE_METHODS: dict[str, PdeMethodSpec] = {
    "explicit_euler": PdeMethodSpec(
        method_id="explicit_euler",
        name="Malha nodal",
        description="Euler explicito com discretizacao espacial em nos da malha e contornos aplicados diretamente.",
    ),
    "ghost_cells": PdeMethodSpec(
        method_id="ghost_cells",
        name="Celulas fantasmas",
        description="Euler explicito em malha centrada nas celulas, com condicoes de contorno impostas via ghost cells.",
    ),
}
