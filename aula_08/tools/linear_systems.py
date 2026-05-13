from __future__ import annotations

import math
import os
import random
import time
import ctypes
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import numpy as np
except ImportError:
    np = None

try:
    from numba import cuda
except ImportError:
    cuda = None


Matrix = list[list[float]]
Vector = list[float]


def _registrar_dlls_cuda_no_windows() -> None:
    if os.name != "nt":
        return

    caminhos: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        base = Path(cuda_path)
        caminhos.extend(
            [
                base / "bin",
                base / "bin" / "x64",
                base / "nvvm" / "bin",
                base / "nvvm" / "bin" / "x64",
            ]
        )

    caminho_padrao = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")
    caminhos.extend(
        [
            caminho_padrao / "bin",
            caminho_padrao / "bin" / "x64",
            caminho_padrao / "nvvm" / "bin",
            caminho_padrao / "nvvm" / "bin" / "x64",
        ]
    )

    runtime_local = Path(__file__).resolve().parent.parent / "cuda_runtime"
    caminhos.insert(0, runtime_local)

    for caminho in caminhos:
        if caminho.exists():
            try:
                os.add_dll_directory(str(caminho))
            except (AttributeError, FileNotFoundError):
                pass


def _precarregar_cudart() -> None:
    if os.name != "nt":
        return

    candidatos = [
        Path(__file__).resolve().parent.parent / "cuda_runtime" / "cudart.dll",
        Path(__file__).resolve().parent.parent / "cuda_runtime" / "cudart64_13.dll",
        Path(__file__).resolve().parent.parent / "cuda_runtime" / "nvvm.dll",
    ]

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        base = Path(cuda_path)
        candidatos.extend(
            [
                base / "bin" / "x64" / "cudart.dll",
                base / "bin" / "x64" / "cudart64_13.dll",
                base / "bin" / "cudart.dll",
                base / "bin" / "cudart64_13.dll",
                base / "nvvm" / "bin" / "x64" / "nvvm.dll",
                base / "nvvm" / "bin" / "x64" / "nvvm64_40_0.dll",
                base / "nvvm" / "bin" / "nvvm.dll",
                base / "nvvm" / "bin" / "nvvm64_40_0.dll",
            ]
        )

    for candidato in candidatos:
        if candidato.exists():
            try:
                ctypes.CDLL(str(candidato))
                return
            except OSError:
                continue


@dataclass
class BenchmarkResult:
    nome: str
    tempo_segundos: float
    residuo_maximo: float
    erro_maximo: float
    disponivel: bool
    observacao: str = ""


def _copiar_sistema(matriz: Matrix, vetor: Vector) -> tuple[Matrix, Vector]:
    return [linha[:] for linha in matriz], vetor[:]


def _trocar_linhas(matriz: Matrix, vetor: Vector, i: int, j: int) -> None:
    matriz[i], matriz[j] = matriz[j], matriz[i]
    vetor[i], vetor[j] = vetor[j], vetor[i]


def _indice_pivo(matriz: Matrix, coluna: int) -> int:
    return max(range(coluna, len(matriz)), key=lambda i: abs(matriz[i][coluna]))


def _retrossubstituicao(matriz: Matrix, vetor: Vector) -> Vector:
    n = len(matriz)
    solucao = [0.0] * n

    for i in range(n - 1, -1, -1):
        soma = sum(matriz[i][j] * solucao[j] for j in range(i + 1, n))
        piv = matriz[i][i]
        if abs(piv) < 1e-12:
            raise ValueError("Sistema singular ou mal condicionado para a tolerancia adotada.")
        solucao[i] = (vetor[i] - soma) / piv

    return solucao


def _substituicao_direta(matriz: Matrix, vetor: Vector) -> Vector:
    n = len(matriz)
    solucao = [0.0] * n

    for i in range(n):
        soma = sum(matriz[i][j] * solucao[j] for j in range(i))
        piv = matriz[i][i]
        if abs(piv) < 1e-12:
            raise ValueError("Sistema singular ou mal condicionado para a tolerancia adotada.")
        solucao[i] = (vetor[i] - soma) / piv

    return solucao


def gauss_serial(matriz: Matrix, vetor: Vector) -> Vector:
    a, b = _copiar_sistema(matriz, vetor)
    n = len(a)

    for k in range(n - 1):
        piv = _indice_pivo(a, k)
        if abs(a[piv][k]) < 1e-12:
            raise ValueError("Nao foi possivel encontrar pivo valido.")
        if piv != k:
            _trocar_linhas(a, b, k, piv)

        for i in range(k + 1, n):
            fator = a[i][k] / a[k][k]
            a[i][k] = 0.0
            for j in range(k + 1, n):
                a[i][j] -= fator * a[k][j]
            b[i] -= fator * b[k]

    return _retrossubstituicao(a, b)


def _aplicar_permutacao(vetor: Vector, permutacoes: list[int]) -> Vector:
    return [vetor[indice] for indice in permutacoes]


def _extrair_lu(fatores: Matrix) -> tuple[Matrix, Matrix]:
    n = len(fatores)
    l = [[0.0] * n for _ in range(n)]
    u = [[0.0] * n for _ in range(n)]

    for i in range(n):
        l[i][i] = 1.0
        for j in range(n):
            if i > j:
                l[i][j] = fatores[i][j]
            else:
                u[i][j] = fatores[i][j]

    return l, u


def decomposicao_lu_serial(matriz: Matrix) -> tuple[Matrix, Matrix, list[int]]:
    fatores = [linha[:] for linha in matriz]
    n = len(fatores)
    permutacoes = list(range(n))

    for k in range(n - 1):
        piv = _indice_pivo(fatores, k)
        if abs(fatores[piv][k]) < 1e-12:
            raise ValueError("Nao foi possivel encontrar pivo valido.")

        if piv != k:
            fatores[k], fatores[piv] = fatores[piv], fatores[k]
            permutacoes[k], permutacoes[piv] = permutacoes[piv], permutacoes[k]

        for i in range(k + 1, n):
            fator = fatores[i][k] / fatores[k][k]
            fatores[i][k] = fator
            for j in range(k + 1, n):
                fatores[i][j] -= fator * fatores[k][j]

    return *_extrair_lu(fatores), permutacoes


def lu_serial(matriz: Matrix, vetor: Vector) -> Vector:
    l, u, permutacoes = decomposicao_lu_serial(matriz)
    b_permutado = _aplicar_permutacao(vetor, permutacoes)
    y = _substituicao_direta(l, b_permutado)
    return _retrossubstituicao(u, y)


def _eliminar_bloco(
    linhas: list[tuple[int, list[float], float]],
    linha_pivo: list[float],
    valor_pivo_b: float,
    coluna_pivo: int,
) -> list[tuple[int, list[float], float]]:
    atualizadas: list[tuple[int, list[float], float]] = []

    for indice, linha, valor_b in linhas:
        fator = linha[coluna_pivo] / linha_pivo[coluna_pivo]
        nova_linha = linha[:]
        nova_linha[coluna_pivo] = 0.0
        for j in range(coluna_pivo + 1, len(nova_linha)):
            nova_linha[j] -= fator * linha_pivo[j]
        atualizadas.append((indice, nova_linha, valor_b - fator * valor_pivo_b))

    return atualizadas


def _eliminar_bloco_lu(
    linhas: list[tuple[int, list[float]]],
    linha_pivo: list[float],
    coluna_pivo: int,
) -> list[tuple[int, list[float]]]:
    atualizadas: list[tuple[int, list[float]]] = []

    for indice, linha in linhas:
        fator = linha[coluna_pivo] / linha_pivo[coluna_pivo]
        nova_linha = linha[:]
        nova_linha[coluna_pivo] = fator
        for j in range(coluna_pivo + 1, len(nova_linha)):
            nova_linha[j] -= fator * linha_pivo[j]
        atualizadas.append((indice, nova_linha))

    return atualizadas


def _dividir_em_blocos(
    a: Matrix,
    b: Vector,
    inicio: int,
    blocos: int,
) -> list[list[tuple[int, list[float], float]]]:
    linhas = [(i, a[i], b[i]) for i in range(inicio, len(a))]
    if not linhas:
        return []

    tamanho_bloco = math.ceil(len(linhas) / blocos)
    return [linhas[i : i + tamanho_bloco] for i in range(0, len(linhas), tamanho_bloco)]


def _dividir_linhas_em_blocos(
    a: Matrix,
    inicio: int,
    blocos: int,
) -> list[list[tuple[int, list[float]]]]:
    linhas = [(i, a[i]) for i in range(inicio, len(a))]
    if not linhas:
        return []

    tamanho_bloco = math.ceil(len(linhas) / blocos)
    return [linhas[i : i + tamanho_bloco] for i in range(0, len(linhas), tamanho_bloco)]


def gauss_parallel_cpu(
    matriz: Matrix,
    vetor: Vector,
    max_workers: int | None = None,
) -> Vector:
    a, b = _copiar_sistema(matriz, vetor)
    n = len(a)
    workers = max_workers or os.cpu_count() or 1

    if workers <= 1 or n < 32:
        return gauss_serial(a, b)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for k in range(n - 1):
            piv = _indice_pivo(a, k)
            if abs(a[piv][k]) < 1e-12:
                raise ValueError("Nao foi possivel encontrar pivo valido.")
            if piv != k:
                _trocar_linhas(a, b, k, piv)

            blocos = _dividir_em_blocos(a, b, k + 1, workers)
            if not blocos:
                continue

            futuros = [
                executor.submit(_eliminar_bloco, bloco, a[k], b[k], k)
                for bloco in blocos
            ]
            for futuro in futuros:
                for indice, nova_linha, novo_b in futuro.result():
                    a[indice] = nova_linha
                    b[indice] = novo_b

    return _retrossubstituicao(a, b)


def decomposicao_lu_parallel_cpu(
    matriz: Matrix,
    max_workers: int | None = None,
) -> tuple[Matrix, Matrix, list[int]]:
    fatores = [linha[:] for linha in matriz]
    n = len(fatores)
    permutacoes = list(range(n))
    workers = max_workers or os.cpu_count() or 1

    if workers <= 1 or n < 32:
        return decomposicao_lu_serial(fatores)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for k in range(n - 1):
            piv = _indice_pivo(fatores, k)
            if abs(fatores[piv][k]) < 1e-12:
                raise ValueError("Nao foi possivel encontrar pivo valido.")

            if piv != k:
                fatores[k], fatores[piv] = fatores[piv], fatores[k]
                permutacoes[k], permutacoes[piv] = permutacoes[piv], permutacoes[k]

            blocos = _dividir_linhas_em_blocos(fatores, k + 1, workers)
            if not blocos:
                continue

            futuros = [
                executor.submit(_eliminar_bloco_lu, bloco, fatores[k], k)
                for bloco in blocos
            ]
            for futuro in futuros:
                for indice, nova_linha in futuro.result():
                    fatores[indice] = nova_linha

    return *_extrair_lu(fatores), permutacoes


def lu_parallel_cpu(
    matriz: Matrix,
    vetor: Vector,
    max_workers: int | None = None,
) -> Vector:
    l, u, permutacoes = decomposicao_lu_parallel_cpu(matriz, max_workers=max_workers)
    b_permutado = _aplicar_permutacao(vetor, permutacoes)
    y = _substituicao_direta(l, b_permutado)
    return _retrossubstituicao(u, y)


if cuda is not None:

    @cuda.jit
    def _kernel_eliminacao(matriz, vetor, linha_pivo, valor_pivo_b, coluna_pivo, n):
        i = cuda.grid(1) + coluna_pivo + 1
        if i >= n:
            return

        fator = matriz[i, coluna_pivo] / linha_pivo[coluna_pivo]
        matriz[i, coluna_pivo] = 0.0
        for j in range(coluna_pivo + 1, n):
            matriz[i, j] -= fator * linha_pivo[j]
        vetor[i] -= fator * valor_pivo_b

    @cuda.jit
    def _kernel_eliminacao_lu(matriz, linha_pivo, coluna_pivo, n):
        i = cuda.grid(1) + coluna_pivo + 1
        if i >= n:
            return

        fator = matriz[i, coluna_pivo] / linha_pivo[coluna_pivo]
        matriz[i, coluna_pivo] = fator
        for j in range(coluna_pivo + 1, n):
            matriz[i, j] -= fator * linha_pivo[j]


def gauss_gpu_cuda(matriz: Matrix, vetor: Vector) -> Vector:
    if np is None or cuda is None:
        raise RuntimeError("CUDA indisponivel: instale numpy e numba com suporte a cuda.")

    _registrar_dlls_cuda_no_windows()
    _precarregar_cudart()

    try:
        gpu_disponivel = cuda.is_available()
    except Exception as exc:
        raise RuntimeError(f"CUDA indisponivel: falha ao inicializar o runtime CUDA ({exc}).") from exc

    if not gpu_disponivel:
        try:
            dispositivos = list(cuda.gpus)
        except Exception:
            dispositivos = []

        if dispositivos:
            raise RuntimeError(
                "CUDA indisponivel: GPU detectada, mas o runtime CUDA nao foi encontrado. "
                "Verifique a instalacao do CUDA Toolkit e se o arquivo cudart.dll esta no PATH."
            )
        raise RuntimeError("CUDA indisponivel: nenhuma GPU compativel foi detectada.")

    a = np.array(matriz, dtype=np.float64)
    b = np.array(vetor, dtype=np.float64)
    n = a.shape[0]

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)

    threads_per_block = 128

    for k in range(n - 1):
        a_host = d_a.copy_to_host()
        b_host = d_b.copy_to_host()

        piv = k + int(np.argmax(np.abs(a_host[k:, k])))
        if abs(a_host[piv, k]) < 1e-12:
            raise ValueError("Nao foi possivel encontrar pivo valido.")

        if piv != k:
            a_host[[k, piv]] = a_host[[piv, k]]
            b_host[[k, piv]] = b_host[[piv, k]]
            d_a = cuda.to_device(a_host)
            d_b = cuda.to_device(b_host)

        linha_pivo = cuda.to_device(a_host[k].copy())
        linhas_abaixo = n - (k + 1)
        if linhas_abaixo <= 0:
            continue

        blocks_per_grid = math.ceil(linhas_abaixo / threads_per_block)
        _kernel_eliminacao[blocks_per_grid, threads_per_block](
            d_a, d_b, linha_pivo, float(b_host[k]), k, n
        )
        cuda.synchronize()

    a_final = d_a.copy_to_host().tolist()
    b_final = d_b.copy_to_host().tolist()
    return _retrossubstituicao(a_final, b_final)


def decomposicao_lu_gpu_cuda(matriz: Matrix) -> tuple[Matrix, Matrix, list[int]]:
    if np is None or cuda is None:
        raise RuntimeError("CUDA indisponivel: instale numpy e numba com suporte a cuda.")

    _registrar_dlls_cuda_no_windows()
    _precarregar_cudart()

    try:
        gpu_disponivel = cuda.is_available()
    except Exception as exc:
        raise RuntimeError(f"CUDA indisponivel: falha ao inicializar o runtime CUDA ({exc}).") from exc

    if not gpu_disponivel:
        try:
            dispositivos = list(cuda.gpus)
        except Exception:
            dispositivos = []

        if dispositivos:
            raise RuntimeError(
                "CUDA indisponivel: GPU detectada, mas o runtime CUDA nao foi encontrado. "
                "Verifique a instalacao do CUDA Toolkit e se o arquivo cudart.dll esta no PATH."
            )
        raise RuntimeError("CUDA indisponivel: nenhuma GPU compativel foi detectada.")

    a = np.array(matriz, dtype=np.float64)
    n = a.shape[0]
    permutacoes = list(range(n))
    d_a = cuda.to_device(a)
    threads_per_block = 128

    for k in range(n - 1):
        a_host = d_a.copy_to_host()
        piv = k + int(np.argmax(np.abs(a_host[k:, k])))
        if abs(a_host[piv, k]) < 1e-12:
            raise ValueError("Nao foi possivel encontrar pivo valido.")

        if piv != k:
            a_host[[k, piv]] = a_host[[piv, k]]
            permutacoes[k], permutacoes[piv] = permutacoes[piv], permutacoes[k]
            d_a = cuda.to_device(a_host)

        linha_pivo = cuda.to_device(a_host[k].copy())
        linhas_abaixo = n - (k + 1)
        if linhas_abaixo <= 0:
            continue

        blocks_per_grid = math.ceil(linhas_abaixo / threads_per_block)
        _kernel_eliminacao_lu[blocks_per_grid, threads_per_block](d_a, linha_pivo, k, n)
        cuda.synchronize()

    fatores = d_a.copy_to_host().tolist()
    return *_extrair_lu(fatores), permutacoes


def lu_gpu_cuda(matriz: Matrix, vetor: Vector) -> Vector:
    l, u, permutacoes = decomposicao_lu_gpu_cuda(matriz)
    b_permutado = _aplicar_permutacao(vetor, permutacoes)
    y = _substituicao_direta(l, b_permutado)
    return _retrossubstituicao(u, y)


def gerar_sistema_linear(tamanho: int, semente: int = 42) -> tuple[Matrix, Vector, Vector]:
    rng = random.Random(semente)
    x_esperado = [float(i + 1) for i in range(tamanho)]
    matriz: Matrix = []

    for i in range(tamanho):
        linha = [rng.uniform(-2.0, 2.0) for _ in range(tamanho)]
        linha[i] += sum(abs(valor) for valor in linha) + 1.0
        matriz.append(linha)

    vetor = [
        sum(matriz[i][j] * x_esperado[j] for j in range(tamanho))
        for i in range(tamanho)
    ]

    return matriz, vetor, x_esperado


def _norma_residuo(matriz: Matrix, vetor: Vector, solucao: Vector) -> float:
    maior = 0.0
    for i, linha in enumerate(matriz):
        ax = sum(linha[j] * solucao[j] for j in range(len(solucao)))
        maior = max(maior, abs(ax - vetor[i]))
    return maior


def _erro_maximo(solucao: Vector, esperado: Vector) -> float:
    return max(abs(solucao[i] - esperado[i]) for i in range(len(solucao)))


def _medir(
    nome: str,
    solver: Callable[[Matrix, Vector], Vector],
    matriz: Matrix,
    vetor: Vector,
    esperado: Vector,
) -> BenchmarkResult:
    try:
        inicio = time.perf_counter()
        solucao = solver(matriz, vetor)
        fim = time.perf_counter()
        return BenchmarkResult(
            nome=nome,
            tempo_segundos=fim - inicio,
            residuo_maximo=_norma_residuo(matriz, vetor, solucao),
            erro_maximo=_erro_maximo(solucao, esperado),
            disponivel=True,
        )
    except Exception as exc:
        return BenchmarkResult(
            nome=nome,
            tempo_segundos=0.0,
            residuo_maximo=float("nan"),
            erro_maximo=float("nan"),
            disponivel=False,
            observacao=str(exc),
        )


def benchmark_solvers(
    tamanho: int = 60,
    semente: int = 42,
    max_workers: int | None = None,
) -> list[BenchmarkResult]:
    matriz, vetor, esperado = gerar_sistema_linear(tamanho, semente)

    resultados = [
        _medir("Gauss serie", gauss_serial, matriz, vetor, esperado),
        _medir(
            "Gauss paralelo CPU",
            lambda a, b: gauss_parallel_cpu(a, b, max_workers=max_workers),
            matriz,
            vetor,
            esperado,
        ),
        _medir("Gauss paralelo GPU (CUDA)", gauss_gpu_cuda, matriz, vetor, esperado),
        _medir("LU serie", lu_serial, matriz, vetor, esperado),
        _medir(
            "LU paralelo CPU",
            lambda a, b: lu_parallel_cpu(a, b, max_workers=max_workers),
            matriz,
            vetor,
            esperado,
        ),
        _medir("LU paralelo GPU (CUDA)", lu_gpu_cuda, matriz, vetor, esperado),
    ]

    return resultados
