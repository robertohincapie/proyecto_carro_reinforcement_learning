"""Piloto fijo que usa solo los 7 sensores. Sirve para validar el entorno y como linea base.

    python -m ejemplos.piloto_heuristico [pista] [--render]
"""

import sys

import numpy as np

import carro_rl  # noqa: F401  (registra CarroPista-v0)
import gymnasium as gym


def piloto(obs: np.ndarray, ganancia: float = 2.5, k_vel: float = 2.0) -> np.ndarray:
    """Gira hacia el lado con mas espacio (angulos positivos = izquierda) y acelera segun el frente libre."""
    izq = obs[4] + obs[5] + 0.5 * obs[6]
    der = obs[2] + obs[1] + 0.5 * obs[0]
    giro = np.clip(ganancia * (izq - der) / (izq + der + 1e-3), -1, 1)
    frente = min(obs[3], obs[2] + 0.2, obs[4] + 0.2)
    # Umbral 0.5: con la traccion base del entorno, fuerza=0 ya avanza; solo hace falta
    # pedir mas cuando el frente esta bien despejado, y frenar cuando esta cerca.
    fuerza = np.clip(k_vel * (frente - 0.5), -1, 1)
    return np.array([fuerza, giro], dtype=np.float32)


if __name__ == "__main__":
    pista = next((a for a in sys.argv[1:] if not a.startswith("--")), "circuito")
    modo = "human" if "--render" in sys.argv else None
    env = gym.make("CarroPista-v0", pista=pista, render_mode=modo)
    obs, info = env.reset(seed=0)
    total, fin = 0.0, False
    while not fin:
        obs, r, term, trunc, info = env.step(piloto(obs))
        total += r
        fin = term or trunc
    print(f"pista={info['pista']} vuelta={info['vuelta_completada']} choque={info['choque']} "
          f"progreso={info['progreso']:.2f} t={info['tiempo']:.1f}s retorno={total:.2f} truncado={trunc}")
    env.close()
