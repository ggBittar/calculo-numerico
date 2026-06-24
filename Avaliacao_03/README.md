# Avaliacao 03 - Metodos Numericos

Este projeto resolve a terceira avaliacao de Metodos Numericos.

O arquivo `main.py` gera:

- simulacoes da conducao termica bidimensional transiente pelo metodo implicito;
- solucao do sistema linear da conducao por Gauss-Seidel;
- simulacoes da equacao da onda 1D por Euler explicito, RK4 explicito e Euler implicito;
- figuras e arquivos CSV em `resultados/`;
- o relatorio em LaTeX `relatorio_avaliacao03.tex`.

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

As figuras usadas pelo relatorio ficam em `resultados/figuras`.
