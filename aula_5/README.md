# aula_5 - Conducao 1D com celulas fantasmas

Aplicacao em Python com interface PyQt6 para simular conducao transiente 1D via diferencas finitas explicitas, usando celulas fantasmas nas duas fronteiras (esquerda e direita).

## Estrutura do projeto

```text
aula_5/
  main.py
  requirements.txt
  thermal_project/
    __init__.py
    models.py
    solver.py
    ui.py
```

- `main.py`: ponto de entrada da aplicacao.
- `thermal_project/models.py`: dataclasses de configuracao e resultado.
- `thermal_project/solver.py`: logica numerica e condicoes de contorno.
- `thermal_project/ui.py`: interface grafica e integracao com o solver.

## Requisitos

- Python 3.11+ (recomendado)
- Dependencias listadas em `requirements.txt`

## Ambiente virtual na raiz do repositorio

Na raiz do repositorio (`calculo-numerico`), crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

No Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

## Instalacao

No diretorio `aula_5`, execute:

```bash
pip install -r requirements.txt
```

## Execucao

No diretorio `aula_5`, execute:

```bash
python main.py
```

## Entradas da interface

- Duas colunas de parametros de contorno:
  - Contorno esquerdo
  - Contorno direito
- Parametros globais:
  - Area `A`
  - Condutividade `k`
  - Temperatura inicial
- Discretizacao:
  - `t0`, `tf`, `Co`
  - `x0`, `xf`, `nx`

## Parametro Co

O passo de tempo nao e' informado diretamente. Ele e' calculado por:

```text
dt = Co * dx^2 / (k * A)
```

Com esquema explicito, use `Co <= 0.5` para estabilidade.

## Saidas

- Resumo numerico (incluindo ghosts esquerdo e direito no tempo selecionado)
- Grafico do perfil `T(x,t)`
- Tabela com indice, coordenada espacial e temperatura
