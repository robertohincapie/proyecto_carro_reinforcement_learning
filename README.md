# proyecto_rl — Carro en pista (Gymnasium)

Entorno de RL inspirado en el reel "Neural Network Driving": un carro con dinámica básica
(tracción/freno, inercia, dirección, derrape) y 7 sensores de distancia a los bordes.

```python
import gymnasium as gym, carro_rl
env = gym.make("CarroPista-v0", pista="circuito")        # "ovalo" | "circuito" | "trebol" | puntos (K,2)
env = gym.make("CarroPista-v0", pista=["ovalo", "trebol"], inicio_aleatorio=True)  # varias pistas
```

- **Acción** `Box(2)` en [-1, 1]: `[fuerza (+ tracción / − freno), dirección]`.
- **Observación** `Box(7)`: distancias de los sensores (−65°, −20°, −10°, 0°, 10°, 20°, 65°) normalizadas por `alcance`.
  `observacion_extendida=True` añade u, v, r (el carro no puede inferir su velocidad solo con los rayos).
- **Estado interno** `env.unwrapped.estado = [x, y, psi, u, v, r, delta]` + `info["s"]`, `info["lateral"]`.
- **Recompensa**: progreso por paso − penalización de tiempo; al completar la vuelta bono ∝ 1/tiempo de vuelta;
  al chocar `-penal_choque`. `terminated` = choque o vuelta; `truncated` = `max_pasos` o 10 s sin mejorar el progreso.
- **Derrape**: modelo de bicicleta con neumáticos saturados (círculo de fricción), ver `carro_rl/dinamica.py`.

```bash
python -m pytest -q
python -m ejemplos.piloto_heuristico circuito --render   # línea base sin aprendizaje
```
