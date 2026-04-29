# Avaliação 02 — Métodos Numéricos com Python + CUDA

Projeto para resolver a **segunda avaliação de Métodos Numéricos** como continuação da Avaliação 01.

O código resolve a condução térmica bidimensional transiente em uma chapa, usando diferenças finitas de segunda ordem no espaço e os métodos explícitos de avanço temporal pedidos:

1. Euler de primeira ordem
2. RK2 — ponto médio
3. RK2 — Euler modificado/Ralston
4. RK2 — Heun
5. RK4 clássico
6. Adams-Bashforth de 2ª ordem
7. Adams-Bashforth de 4ª ordem

O projeto tenta usar **CUDA via CuPy** automaticamente. Se CuPy/CUDA não estiver disponível, usa NumPy na CPU para permitir depuração.

## Estrutura

```text
avaliacao02_cuda_python/
├── avaliacao02/
│   ├── backend.py              # escolhe CuPy/CUDA ou NumPy
│   ├── config.py               # propriedades, geometria, contornos e parâmetros
│   ├── spatial_operator.py     # operador de diferenças finitas 2D
│   ├── time_methods.py         # Euler, RK2, RK4, AB2, AB4
│   ├── solver.py               # rotina de simulação
│   └── plots.py                # gráficos pedidos
├── scripts/
│   ├── run_all.py              # executa todos os casos da avaliação
│   └── run_single.py           # executa um caso específico
├── resultados/
│   ├── dados/                  # CSVs gerados
│   └── figuras/                # PNGs gerados
├── requirements.txt
└── README.md
```

## Instalação

Crie um ambiente Python e instale as dependências gerais:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell/CMD
pip install -r requirements.txt
```

Para usar GPU NVIDIA, instale o CuPy compatível com sua versão do CUDA. Exemplos:

```bash
pip install cupy-cuda12x
```

ou, para CUDA 11:

```bash
pip install cupy-cuda11x
```

## Como executar tudo

A partir da pasta raiz do projeto:

```bash
python scripts/run_all.py --cuda
```

Para forçar CPU:

```bash
python scripts/run_all.py --cpu
```

Para controlar a paralelização em CPU:

```bash
python scripts/run_all.py --cpu --workers 6
```

Observação: em `--cpu`, os casos independentes são distribuídos entre múltiplos processos.
Em `--cuda`, a execução continua sequencial para evitar disputa pelo mesmo dispositivo GPU.
O executor também aplica limites conservadores de estabilidade por método e ajusta
automaticamente `C` quando necessário.

## Como executar um único caso

```bash
python scripts/run_single.py --method rk4 --N 20 --C 0.25 --cuda
```

Métodos aceitos:

```text
euler
rk2_ponto_medio
rk2_euler_modificado
rk2_heun
rk4
ab2
ab4
```

## Observações importantes para o relatório

- A Avaliação 02 pede os 7 métodos explícitos aplicados ao algoritmo da Avaliação 01.
- O passo de tempo é calculado por:

```text
dt = C * min(dx, dy)^2 / alpha
```

- Para Euler explícito, em malha uniforme 2D, recomenda-se `C <= 0,25`.
- Para RK4, o projeto usa um teto conservador de `C <= 0,34`.
- Para AB2, o projeto usa um teto conservador de `C <= 0,12`.
- Para AB4, o projeto usa um teto conservador de `C <= 0,03`.
- Os testes com `C = 0,5`, `0,25` e `0,125` são executados para RK4 e AB4 conforme o enunciado.
  Quando esses valores excedem a faixa estável do método, o projeto reescala a
  sequência automaticamente preservando as proporções relativas.
- As propriedades do alumínio estão em `avaliacao02/config.py` e podem ser alteradas caso seja necessário usar uma tabela específica.

## Saídas

Após `run_all.py`, são gerados:

- `resultados/dados/resultados_consolidados.csv`
- CSVs separados por método/malha/C
- gráficos `.png` para evolução temporal em:
  - `x=Lx/2, y=Ly/2`
  - `x=Lx/4, y=Ly/4`

## Programas separados por método

Além do executor geral, há um script individual para cada método, atendendo à exigência de programas separados:

```bash
python scripts/executar_euler.py
python scripts/executar_rk2_ponto_medio.py
python scripts/executar_rk2_euler_modificado.py
python scripts/executar_rk2_heun.py
python scripts/executar_rk4.py
python scripts/executar_adams_bashforth_2.py
python scripts/executar_adams_bashforth_4.py
```
