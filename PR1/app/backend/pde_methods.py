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
        name="Euler explícito",
        description="Avanco temporal por Euler e discretizacao espacial por diferencas finitas centrais.",
    ),
}
