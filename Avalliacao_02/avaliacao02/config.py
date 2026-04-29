"""Parâmetros físicos, geométricos e numéricos do problema.

Os valores geométricos e de contorno seguem o enunciado da Avaliação 1.
As propriedades do alumínio puro foram deixadas editáveis. Caso o professor
forneça outra tabela/propriedade, basta alterar `MaterialProperties`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlateGeometry:
    """Geometria da chapa."""

    Lx: float = 1.0          # comprimento em x [m]
    Ly: float = 1.0          # comprimento em y [m]
    thickness: float = 5e-3  # espessura [m]; não entra na EDP 2D sem geração volumétrica


@dataclass(frozen=True)
class MaterialProperties:
    """Propriedades do alumínio puro adotadas no exemplo.

    k, rho e cp podem ser alterados conforme a tabela indicada pelo professor.
    """

    k: float = 237.0       # condutividade térmica [W/(m.K)]
    rho: float = 2700.0    # massa específica [kg/m³]
    cp: float = 900.0      # calor específico [J/(kg.K)]

    @property
    def alpha(self) -> float:
        """Difusividade térmica alpha = k/(rho*cp) [m²/s]."""
        return self.k / (self.rho * self.cp)


@dataclass(frozen=True)
class BoundaryConditions:
    """Condições de contorno e inicial."""

    T_left: float = 400.0      # x=0: temperatura imposta [K]
    T_initial: float = 300.0   # temperatura inicial [K]

    # y=0: convecção
    h_bottom: float = 100.0    # [W/(m².K)]
    Tinf_bottom: float = 350.0 # [K]

    # y=Ly: convecção + radiação
    h_top: float = 200.0       # [W/(m².K)]
    Tinf_top: float = 280.0    # [K]
    T_sur: float = 300.0       # temperatura da vizinhança radiativa [K]
    emissivity: float = 0.8    # emissividade [-]
    sigma: float = 5.670374419e-8  # Stefan-Boltzmann [W/(m².K⁴)]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuração geral das simulações."""

    geometry: PlateGeometry = field(default_factory=PlateGeometry)
    material: MaterialProperties = field(default_factory=MaterialProperties)
    bc: BoundaryConditions = field(default_factory=BoundaryConditions)
    t_final: float = 4.0 * 3600.0   # 4 h em segundos
    default_C: float = 0.25         # coeficiente usado em Δt = C*min(dx,dy)^2/alpha
    save_every_seconds: float = 60.0 # intervalo aproximado para armazenar histórico


# Malhas e coeficientes pedidos na Avaliação 02.
MESHES: tuple[int, ...] = (10, 20, 40, 80)
STABILITY_COEFFICIENTS: tuple[float, ...] = (0.5, 0.25, 0.125)
