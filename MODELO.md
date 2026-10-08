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
| `velocidad_inicial` | Velocidad longitudinal (m/s) con la que arranca el carro en cada reset; 5 por defecto. |
| `aceleracion_base` | Traccion real (en la escala de `fuerza` de la dinamica, [-1, 1]) que recibe el carro cuando la accion de fuerza es 0; 0.5 por defecto. Evita que el carro se quede quieto si el agente no hace nada. |
| `observar_dinamica` | `True` (por defecto): la observación incluye velocidades. `False`: solo los 7 rayos. |
| `alcance` | Distancia máxima de los sensores (60 m). |
| `max_pasos`, `tiempo_sin_progreso` | Criterios de truncamiento. |
| `peso_velocidad`, `velocidad_ref` | Constantes de la recompensa. |
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

- **fuerza:** no se pasa directo a la dinámica. Se reescala con `aceleracion_base` (0.5 por defecto)
  de forma que `fuerza = 0` ya empuja al carro con una tracción base de 0.5 (en vez de 0), así el carro
  no se queda quieto si el agente no hace nada. `fuerza = 1` pide tracción máxima, `fuerza = -1` frena a
  fondo, y entre 0 y -1 se pasa de la tracción base al frenado total. Es un solo valor, así que nunca hay
  tracción y freno a la vez. La reescala es lineal a tramos (continua en 0):
  ```
  fuerza_real = aceleracion_base + (1 − aceleracion_base) · fuerza        si fuerza ≥ 0
  fuerza_real = aceleracion_base + (1 + aceleracion_base) · fuerza        si fuerza < 0
  ```
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
r = peso_velocidad · (u / velocidad_ref) · dt
```

Solo premia la velocidad longitudinal `u` (el carro no tiene marcha atrás, `u ≥ 0`) relativa a
`velocidad_ref`. Con los valores por defecto (`peso_velocidad = 1`, `velocidad_ref = 20 m/s`), ir
justo a la velocidad de referencia da +1 por segundo; al doble de `velocidad_ref` da +2 por segundo,
y así proporcionalmente.

No hay ningún otro término ni penalización: ni por progreso, ni por ir lento, ni por tiempo, ni por
chocar. Un choque (alguna esquina del carro sale de la calzada) no resta nada; simplemente termina el
episodio (`terminated`), así que la recompensa deja de poder seguir creciendo. El truncamiento por
`max_pasos` o por estancamiento (`tiempo_sin_progreso`) tampoco suma ni resta nada, solo corta el
episodio.

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
