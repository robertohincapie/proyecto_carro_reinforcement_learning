# proyecto_rl — Carro en pista (Gymnasium)

Entorno de RL inspirado en el reel "Neural Network Driving": un carro con dinámica básica
(tracción/freno, inercia, dirección, derrape) y 7 sensores de distancia a los bordes.

```python
import gymnasium as gym, carro_rl
env = gym.make("CarroPista-v0", pista="circuito")        # "ovalo" | "circuito" | "trebol" | puntos (K,2)
env = gym.make("CarroPista-v0", pista=["ovalo", "trebol"], inicio_aleatorio=True)  # varias pistas
```

- **Acción** `Box(2)` en [-1, 1]: `[fuerza (+ tracción / − freno), dirección]`.
- **Observación** `Box(10)`: 7 distancias de los sensores (−65°, −20°, −10°, 0°, 10°, 20°, 65°) normalizadas por `alcance`,
  más la dinámica del carro: velocidad longitudinal u/40, lateral v/10 y de giro r/2 (recortadas a [−1, 1]).
  `observar_dinamica=False` deja solo los 7 rayos.
- **Estado interno** `env.unwrapped.estado = [x, y, psi, u, v, r, delta]` + `info["s"]`, `info["lateral"]`.
- **Recompensa**: progreso por paso + premio ∝ velocidad longitudinal u − penalización de tiempo; al completar la vuelta bono ∝ 1/tiempo de vuelta;
  al chocar `-penal_choque`. `terminated` = choque o vuelta; `truncated` = `max_pasos` o 10 s sin mejorar el progreso.
- **Derrape**: modelo de bicicleta con neumáticos saturados (círculo de fricción), ver `carro_rl/dinamica.py`.

Se gestiona con [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                   # crea .venv e instala dependencias (según uv.lock)
uv run pytest -q
uv run python -m ejemplos.piloto_heuristico circuito --render   # línea base sin aprendizaje
uv add stable-baselines3                  # añadir dependencias
```

Detalle del modelo (pista, dinámica, observación, acciones y recompensa): [MODELO.md](MODELO.md).
