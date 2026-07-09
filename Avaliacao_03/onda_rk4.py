import numpy as np

from main import DATA, WaveConfig, ensure_dirs, simulate_wave


if __name__ == "__main__":
    ensure_dirs()
    cfg = WaveConfig()
    result = simulate_wave("rk4", cfg)
    output = DATA / "onda_rk4.csv"
    np.savetxt(
        output,
        np.column_stack([result["times"], result["y_center"], result["v_center"], result["errors"]]),
        delimiter=",",
        header="tempo,y_centro,v_centro,erro_l2",
        comments="",
    )
    print(f"Resultado do RK4 salvo em {output}")
