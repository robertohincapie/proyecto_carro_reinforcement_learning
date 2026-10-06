# Modelo del entorno `CarroPista-v0`

`CarroPistaEnv` (en `carro_rl/entorno.py`) simula un carro en una pista cerrada. En cada paso de control
(`dt = 0.05 s`, 20 Hz) integra la dinámica con 5 subpasos de 0.01 s, calcula los sensores, detecta choque
y entrega la recompensa.

## Cómo se carga

```python
import gymnasium as gym
import carro_rl                       # al importarlo se registra "CarroPista-v0"

env = gym.make("CarroPista-v0", pista="circuito", render_mode="human")
obs, info = env.reset(seed=0)
obs, r, terminated, truncated, info = env.step(env.action_space.sample())
```

También se puede instanciar directo: `from carro_rl import CarroPistaEnv`.

| Parámetro | Qué hace |
|---|---|
| `pista` | Nombre (`"ovalo"`, `"circuito"`, `"trebol"`), una `Pista`, un arreglo de puntos de control (K, 2), o una **lista** de pistas (cada `reset` elige una al azar). |
| `inicio_aleatorio` | Si es `True`, el carro arranca en un punto aleatorio de la pista. |
| `observar_dinamica` | `True` (por defecto): la observación incluye velocidades. `False`: solo los 7 rayos. |
| `alcance` | Distancia máxima de los sensores (60 m). |
| `max_pasos`, `tiempo_sin_progreso` | Criterios de truncamiento. |
| `peso_progreso`, `penal_tiempo`, `bono_vuelta`, `penal_choque`, `penal_truncado`, `velocidad_ref` | Constantes de la recompensa. |
| `carro` | Un `ParametrosCarro` con masa, agarre, fuerzas y límites de dirección. |

## Pista

Cada pista es una línea central cerrada (un spline que pasa por los puntos de control) con un ancho de 12 m,
remuestreada en puntos equiespaciados. De ahí salen los dos bordes, que se usan para los sensores y para
detectar choques. Si la curvatura es tan cerrada que los bordes se cruzarían, el constructor lanza un error.

## Dinámica del carro (`carro_rl/dinamica.py`)

Modelo de bicicleta con neumáticos que se saturan.

- **Estado interno** (`env.unwrapped.estado`): `[x, y, psi, u, v, r, delta]`: posición, rumbo, velocidad
  hacia adelante, velocidad lateral, velocidad de giro y ángulo real de las ruedas delanteras.
- **Derrape:** la fuerza lateral de cada eje crece con el ángulo de deslizamiento, pero se satura con el
  agarre (`mu`). La tracción o el frenado le quitan agarre lateral al eje correspondiente (círculo de
  fricción). Por eso entrar rápido a una curva, o acelerar o frenar a fondo en ella, produce derrape.
- **Tracción:** trasera, hasta 6000 N.
- **Freno:** hasta 14000 N, repartido 60 % delante y 40 % detrás.
- **Marcha atrás:** no existe; `u` nunca baja de 0.
- **Dirección:** el volante sigue al comando con límite de velocidad (1.5 rad/s) y máximo de 0.5 rad.

## Acción

`Box(2)` en [-1, 1]: `[fuerza, dirección]`.

- **fuerza:** positiva es tracción, negativa es freno. Es un solo valor, así que nunca hay ambas a la vez.
- **dirección:** ángulo objetivo del volante, escalado a ±0.5 rad.

## Observación

`Box(10)`, `float32`:

| Índices | Contenido | Normalización |
|---|---|---|
| 0–6 | Distancia del sensor al borde, con ángulos −65°, −20°, −10°, 0°, 10°, 20°, 65° respecto al rumbo | ÷ `alcance`, en [0, 1] |
| 7 | Velocidad longitudinal `u` | ÷ 40, recortada a [−1, 1] |
| 8 | Velocidad lateral `v` | ÷ 10, recortada a [−1, 1] |
| 9 | Velocidad de giro `r` | ÷ 2, recortada a [−1, 1] |

Los ángulos positivos apuntan a la izquierda. El agente no ve su posición en la pista; eso queda en `info`.

## Recompensa

En cada paso:

```
r = peso_progreso · (avance / longitud_pista)  −  penal_tiempo · dt
```

Con los valores por defecto, una vuelta completa suma +10 por progreso y cada segundo resta 0.05.

| Evento | Efecto |
|---|---|
| Vuelta completada | `+ bono_vuelta · t_ref / t_vuelta` (20 por defecto), con `t_ref = longitud / velocidad_ref` y `velocidad_ref = 20 m/s`. A 20 m/s promedio el bono es 20; con el doble de tiempo, 10. |
| Choque | `− penal_choque` (10) |
| Truncamiento | `+ penal_truncado` (0 por defecto) |

El bono de vuelta solo aparece si el agente completa una vuelta; los términos de progreso y de tiempo son los
que guían el aprendizaje desde el inicio.

## Fin de episodio

- **`terminated`:** choque (alguna de las 4 esquinas del carro sale de la calzada) o vuelta completada
  (el progreso acumulado alcanza la longitud de la pista).
- **`truncated`:** se llega a `max_pasos` (3000, es decir 150 s) o pasan 10 s sin mejorar el mejor progreso.
  La segunda condición también corta el caso de ir en sentido contrario o quedarse quieto.

## `info`

- `s` (posición a lo largo de la pista) y `lateral` (desplazamiento respecto al centro).
- `progreso` (fracción de la vuelta), `velocidad`, `derrape` (ángulo de deslizamiento).
- `tiempo`, `choque`, `vuelta_completada`, `tiempo_vuelta`, `pista`.

## Advertencia

Los valores del carro (masa, agarre, fuerzas) son estimaciones razonables, no medidas de nada. Para otro
manejo, cambia `ParametrosCarro`.
