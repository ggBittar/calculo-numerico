import argparse

from tools.linear_systems import benchmark_solvers


def imprimir_resultados(tamanho: int = 120, max_workers: int | None = None) -> None:
    resultados = benchmark_solvers(tamanho=tamanho, max_workers=max_workers)

    print(f"Benchmark de Gauss e LU para sistema {tamanho}x{tamanho}")
    print("-" * 78)
    print(f"{'Metodo':28} {'Tempo (s)':>12} {'Residuo':>16} {'Erro':>16}  Observacao")
    print("-" * 78)

    for resultado in resultados:
        tempo = f"{resultado.tempo_segundos:.6f}" if resultado.disponivel else "-"
        residuo = f"{resultado.residuo_maximo:.3e}" if resultado.disponivel else "-"
        erro = f"{resultado.erro_maximo:.3e}" if resultado.disponivel else "-"
        observacao = resultado.observacao or "ok"
        print(f"{resultado.nome:28} {tempo:>12} {residuo:>16} {erro:>16}  {observacao}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark dos metodos de Gauss e LU.")
    parser.add_argument(
        "--tamanho",
        type=int,
        default=120,
        help="Tamanho n do sistema linear n x n.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=15,
        help="Quantidade maxima de processos na versao paralela em CPU.",
    )
    args = parser.parse_args()
    imprimir_resultados(tamanho=args.tamanho, max_workers=args.workers)
