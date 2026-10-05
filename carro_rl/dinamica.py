"""Modelo dinamico de bicicleta con neumaticos saturados (circulo de friccion).

Estado: [x, y, psi, u, v, r, delta]
  x, y   posicion en el mundo (m)
  psi    rumbo (rad)
  u, v   velocidad longitudinal y lateral en ejes del carro (m/s)
  r      velocidad de guinada (rad/s)
  delta  angulo real de las ruedas delanteras (rad)

Cuando se pide mas fuerza de la que el neumatico puede dar (curva a mucha velocidad,
acelerar o frenar a fondo en curva) la fuerza lateral se satura y el carro derrapa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class ParametrosCarro:
    masa: float = 1200.0
    inercia: float = 2500.0
    a: float = 1.3  # CG -> eje delantero
    b: float = 1.3  # CG -> eje trasero
    largo: float = 4.2
    ancho: float = 1.8
    rigidez_curva: float = 90000.0  # N/rad por eje
    mu: float = 1.0
    fuerza_motor: float = 6000.0  # N, traccion trasera
    fuerza_freno: float = 14000.0  # N, 60 % delante / 40 % detras
    reparto_freno_delantero: float = 0.6
    c_aire: float = 2.5  # N/(m/s)^2
    c_rodadura: float = 0.015
    max_direccion: float = 0.5  # rad
    vel_direccion: float = 1.5  # rad/s


def _sat(f_pedida: float, f_max: float) -> float:
    return f_max * math.tanh(f_pedida / f_max) if f_max > 1e-6 else 0.0


def paso(estado: np.ndarray, fuerza: float, direccion: float, p: ParametrosCarro, dt: float) -> np.ndarray:
    """Avanza dt segundos. fuerza en [-1, 1] (+ traccion, - freno); direccion en [-1, 1]."""
    x, y, psi, u, v, r, delta = (float(e) for e in estado)
    fz = p.masa * G * 0.5  # a == b -> reparto 50/50 (sin transferencia de carga)
    objetivo = float(np.clip(direccion, -1.0, 1.0)) * p.max_direccion
    delta += float(np.clip(objetivo - delta, -p.vel_direccion * dt, p.vel_direccion * dt))

    cmd = float(np.clip(fuerza, -1.0, 1.0))
    if cmd >= 0.0:
        fx_del, fx_tra = 0.0, cmd * p.fuerza_motor
    else:
        fb = -cmd * p.fuerza_freno * math.tanh(u / 0.5)
        fx_del, fx_tra = -fb * p.reparto_freno_delantero, -fb * (1 - p.reparto_freno_delantero)
    lim = p.mu * fz
    fx_del, fx_tra = max(-lim, min(lim, fx_del)), max(-lim, min(lim, fx_tra))

    u_ef = max(u, 3.0)  # evita la singularidad del angulo de deslizamiento a baja velocidad
    alfa_d = math.atan2(v + p.a * r, u_ef) - delta
    alfa_t = math.atan2(v - p.b * r, u_ef)
    fy_d = -_sat(p.rigidez_curva * alfa_d, math.sqrt(max(lim**2 - fx_del**2, 0.0)))
    fy_t = -_sat(p.rigidez_curva * alfa_t, math.sqrt(max(lim**2 - fx_tra**2, 0.0)))

    resist = p.c_aire * u * abs(u) + p.c_rodadura * p.masa * G * math.tanh(u / 0.5)
    cd, sd = math.cos(delta), math.sin(delta)
    du = ((fx_del * cd - fy_d * sd) + fx_tra - resist) / p.masa + v * r
    dv = ((fy_d * cd + fx_del * sd) + fy_t) / p.masa - u * r
    dr = (p.a * (fy_d * cd + fx_del * sd) - p.b * fy_t) / p.inercia

    u += du * dt
    v += dv * dt
    r += dr * dt
    if u < 0.0:  # sin marcha atras
        u = 0.0
    psi += r * dt
    x += (u * math.cos(psi) - v * math.sin(psi)) * dt
    y += (u * math.sin(psi) + v * math.cos(psi)) * dt
    return np.array([x, y, psi, u, v, r, delta])


def esquinas(estado: np.ndarray, p: ParametrosCarro) -> np.ndarray:
    """Las 4 esquinas del carro (4, 2) en el mundo."""
    x, y, psi = estado[0], estado[1], estado[2]
    local = np.array(
        [[p.largo / 2, p.ancho / 2], [p.largo / 2, -p.ancho / 2], [-p.largo / 2, -p.ancho / 2], [-p.largo / 2, p.ancho / 2]]
    )
    c, s = math.cos(psi), math.sin(psi)
    return np.stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], axis=1)
