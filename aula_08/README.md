# Aula 08 - Solucao de Sistemas Lineares

Este diretorio contem uma implementacao didatica para resolucao de sistemas lineares usando:

- eliminacao de Gauss
- decomposicao LU
- execucao em serie
- paralelizacao em CPU
- execucao em GPU com CUDA
- comparacao de tempos, residuo numerico e erro da solucao

O objetivo principal deste material e comparar diferentes formas de resolver o mesmo sistema linear e observar o custo computacional de cada abordagem.

## Estrutura

```text
aula_08/
|-- main.py
|-- requirements.txt
|-- README.md
|-- tools/
|   |-- __init__.py
|   `-- linear_systems.py
`-- cuda_runtime/
```

### Arquivos principais

- `main.py`
  Executa o benchmark e imprime a tabela com os resultados.

- `tools/linear_systems.py`
  Implementa os metodos numericos, as versoes paralelas e as rotinas auxiliares.

- `requirements.txt`
  Lista as dependencias Python do projeto.

- `cuda_runtime/`
  Diretorio de apoio usado no Windows para contornar casos em que o `numba` nao encontra automaticamente DLLs do CUDA e do NVVM.

## Metodos implementados

### 1. Metodo de Gauss

As seguintes versoes foram implementadas:

- `gauss_serial`
  Eliminacao de Gauss tradicional com pivotamento parcial e retrossubstituicao.

- `gauss_parallel_cpu`
  Divide a eliminacao das linhas abaixo do pivo em blocos e distribui o trabalho entre processos via `ProcessPoolExecutor`.

- `gauss_gpu_cuda`
  Usa kernel CUDA com `numba.cuda` para atualizar linhas da matriz em paralelo na GPU.

### 2. Decomposicao LU

As seguintes versoes foram implementadas:

- `lu_serial`
  Decompoe a matriz em `L` e `U`, aplica permutacoes de pivotamento parcial e resolve o sistema por substituicao direta e retrossubstituicao.

- `lu_parallel_cpu`
  Faz a atualizacao das linhas em paralelo na etapa de decomposicao.

- `lu_gpu_cuda`
  Usa kernel CUDA para realizar a etapa de eliminacao que gera os fatores da decomposicao LU.

## Como o benchmark funciona

O benchmark gera automaticamente um sistema linear quadrado `A x = b` com:

- matriz diagonalmente dominante
- solucao esperada conhecida
- comparacao entre a solucao calculada e a solucao exata

Para cada metodo sao medidos:

- `Tempo (s)`: tempo total de execucao
- `Residuo`: valor maximo de `|A x - b|`
- `Erro`: erro maximo entre a solucao calculada e a solucao esperada
- `Observacao`: status da execucao ou mensagem de erro

## Dependencias

Instale as dependencias com:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ou, se estiver usando outro Python:

```powershell
python -m pip install -r requirements.txt
```

Conteudo atual de `requirements.txt`:

```txt
numpy>=1.26
numba>=0.59
```

### Papel de cada dependencia

- `numpy`
  Usado principalmente nas rotinas de GPU e nas conversoes de matriz para arrays numericos.

- `numba`
  Usado para compilar e executar kernels CUDA na GPU.

### Biblioteca padrao

A paralelizacao em CPU usa apenas a biblioteca padrao do Python:

- `concurrent.futures`
- `ProcessPoolExecutor`

Por isso, a versao paralela em CPU nao depende de bibliotecas extras alem do proprio Python.

## Como executar

No terminal, dentro de `aula_08`, rode:

```powershell
.\.venv\Scripts\python.exe main.py
```

Ou a partir da raiz do repositorio:

```powershell
.\aula_08\.venv\Scripts\python.exe aula_08\main.py
```

### Argumentos disponiveis

- `--tamanho`
  Define o tamanho `n` do sistema `n x n`.

- `--workers`
  Define a quantidade maxima de processos da versao paralela em CPU.

Exemplo:

```powershell
.\aula_08\.venv\Scripts\python.exe aula_08\main.py --tamanho 200 --workers 8
```

## Exemplo de saida

```text
Benchmark de Gauss e LU para sistema 20x20
------------------------------------------------------------------------------
Metodo                          Tempo (s)          Residuo             Erro  Observacao
------------------------------------------------------------------------------
Gauss serie                      0.000179        1.137e-13        3.553e-15  ok
Gauss paralelo CPU               0.000171        1.137e-13        3.553e-15  ok
Gauss paralelo GPU (CUDA)        0.275907        1.137e-13        7.105e-15  ok
LU serie                         0.000223        1.705e-13        1.066e-14  ok
LU paralelo CPU                  0.000210        1.705e-13        1.066e-14  ok
LU paralelo GPU (CUDA)           0.082431        1.137e-13        3.553e-15  ok
```

## Interpretacao dos resultados

Nem sempre a versao paralela sera mais rapida.

### Serie

Para sistemas pequenos, as versoes em serie costumam ser mais eficientes porque nao ha custo de sincronizacao, criacao de processos ou transferencia para GPU.

### Paralelo em CPU

O metodo paralelo em CPU pode ficar mais lento em tamanhos pequenos e medios por causa do overhead de:

- criar processos
- copiar dados entre processos
- sincronizar resultados

Para matrizes maiores, esse custo pode passar a compensar.

### Paralelo em GPU

O metodo em GPU pode ser excelente para tamanhos grandes, mas ha custos adicionais:

- copia da matriz da memoria principal para a GPU
- compilacao do kernel CUDA
- sincronizacao entre CPU e GPU

Por isso, para sistemas pequenos, e normal a GPU ser mais lenta do que a CPU.

## CUDA no Windows

Para a versao GPU funcionar no Windows com `numba`, o ambiente precisa estar consistente em tres niveis:

1. GPU NVIDIA compativel
2. driver NVIDIA atualizado
3. CUDA Toolkit compativel com o driver

### Problemas encontrados durante o desenvolvimento

Durante a configuracao deste projeto apareceram dois problemas comuns:

- o `numba` nao encontrava `cudart.dll`
- o `numba` nao encontrava `nvvm.dll`

Em algumas instalacoes recentes do CUDA no Windows, as DLLs podem estar presentes apenas com nomes versionados, por exemplo:

- `cudart64_13.dll`
- `nvvm64_40_0.dll`

Por isso o projeto usa:

- registro manual de diretorios de DLL
- pre-carregamento de DLLs
- fallback local em `cuda_runtime/`

### Erro de versao PTX

Outro erro comum e:

```text
CUDA_ERROR_UNSUPPORTED_PTX_VERSION
Unsupported .version 9.2; current version is '9.0'
```

Esse erro indica incompatibilidade entre:

- a versao do Toolkit CUDA usada pelo ambiente
- a versao suportada pelo driver NVIDIA instalado

Se isso acontecer, ha duas solucoes comuns:

- atualizar o driver NVIDIA
- instalar uma versao do CUDA Toolkit compativel com o driver

## Funcoes principais do modulo

As funcoes centrais em `tools/linear_systems.py` incluem:

- `gauss_serial`
- `gauss_parallel_cpu`
- `gauss_gpu_cuda`
- `decomposicao_lu_serial`
- `decomposicao_lu_parallel_cpu`
- `decomposicao_lu_gpu_cuda`
- `lu_serial`
- `lu_parallel_cpu`
- `lu_gpu_cuda`
- `benchmark_solvers`

## Observacoes numericas

- As implementacoes usam pivotamento parcial para melhorar estabilidade numerica.
- O codigo rejeita sistemas singulares ou mal condicionados com pivos muito proximos de zero.
- O benchmark usa uma matriz diagonalmente dominante para reduzir a chance de instabilidade numerica no exemplo padrao.

## Limitacoes atuais

- A implementacao foi pensada com foco didatico, nao como biblioteca de alto desempenho.
- A versao paralela em CPU usa copia de dados entre processos, o que aumenta overhead.
- A versao em GPU ainda depende fortemente da configuracao local de CUDA e driver.
- Para matrizes pequenas, os tempos da GPU podem parecer piores do que os da CPU.

## Sugestoes de experimentos

- Testar diferentes valores de `--tamanho`
- Comparar o impacto de `--workers`
- Medir quando a GPU comeca a compensar
- Comparar Gauss e LU em matrizes maiores
- Registrar graficos de tempo por tamanho do sistema

## Resumo

Este projeto serve como base para estudar:

- resolucao de sistemas lineares
- eliminacao de Gauss
- decomposicao LU
- paralelizacao em CPU
- aceleracao com GPU usando CUDA
- comparacao de desempenho entre abordagens numericas

Ele foi estruturado para ser simples de executar, facil de expandir e adequado para uso academico e experimental.
