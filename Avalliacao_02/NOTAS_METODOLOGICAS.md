# Notas metodológicas para usar no relatório

## EDP resolvida

A condução térmica bidimensional transiente com condutividade constante é escrita como:

```text
∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²)
α = k/(ρ cp)
```

## Discretização espacial

Foram usadas diferenças finitas centradas de segunda ordem nos termos espaciais.

Para pontos internos:

```text
dT[i,j]/dt = α [ (T[i+1,j] - 2T[i,j] + T[i-1,j])/dx²
              + (T[i,j+1] - 2T[i,j] + T[i,j-1])/dy² ]
```

## Contornos

- `x=0`: temperatura imposta `T=400 K`.
- `x=Lx`: adiabático, aproximado por `T[-1,:] = T[-2,:]`.
- `y=0`: convecção, incorporada por ponto fantasma.
- `y=Ly`: convecção + radiação, incorporada por ponto fantasma.

## Avanço temporal

Os métodos implementados são:

- Euler explícito
- RK2 ponto médio
- RK2 Euler modificado/Ralston
- RK2 Heun
- RK4 clássico
- Adams-Bashforth 2
- Adams-Bashforth 4

Os métodos Adams-Bashforth são inicializados com RK4 até haver histórico suficiente.

## Passo temporal

O passo temporal é calculado como:

```text
Δt = C min(Δx, Δy)^2 / α
```

O código ajusta levemente `Δt` para que o tempo final seja exatamente `4 h = 14400 s`.
