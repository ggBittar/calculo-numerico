# aula_4

Aplicacao desktop em Python para experimentos de calculo numerico com:

- estimativa de derivadas por diferencas finitas;
- estimativa de funcao com base em EDPs usando Euler explicito.

## Requisitos

- Python 3.11+

## Preparacao do ambiente

```powershell
cd aula_4
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Isso tambem compila os kernels de Cython (`app/backend/pde_kernels.pyx`) usados no solver de EDP.

## Execucao

Opcao 1:

```powershell
cd aula_4
python main.py
```

Opcao 2:

```powershell
cd aula_4
aula_4
```

## Estrutura

- `app/backend`: cadastros de funcoes, metodos e problemas de EDP.
- `app/ui`: interface grafica.
- `main.py`: inicializacao simples para execucao local.
