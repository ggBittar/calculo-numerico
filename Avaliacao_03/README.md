# Avaliação 03 - Métodos Numéricos

Este projeto resolve a terceira avaliação de Métodos Numéricos.

O arquivo `main.py` gera:

- simulações da condução térmica bidimensional transiente pelo método implícito;
- solução do sistema linear da condução por Gauss-Seidel;
- simulações da equação da onda 1D por Euler explícito, RK4 explícito e Euler implícito;
- figuras e arquivos de resumo em `resultados/`;
- o relatório em LaTeX `relatorio_avaliacao03.tex`.

Também há três executores separados para a questão da onda:

```bash
python onda_euler_explicito.py
python onda_rk4.py
python onda_euler_implicito.py
```

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Como gerar o PDF

Depois de executar o Python, compile:

```bash
pdflatex -interaction=nonstopmode -halt-on-error relatorio_avaliacao03.tex
pdflatex -interaction=nonstopmode -halt-on-error relatorio_avaliacao03.tex
```

As figuras usadas pelo relatório ficam em `resultados/figuras`.
