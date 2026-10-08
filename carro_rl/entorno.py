"""Entorno Gymnasium: carro con dinamica basica que debe dar una vuelta a una pista."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import dinamica
from .dinamica import ParametrosCarro
from .pistas import Pista, crear_pista

ANGULOS_SENSORES = np.deg2rad([-65.0, -20.0, -10.0, 0.0, 10.0, 20.0, 65.0])


class CarroPistaEnv(gym.Env):
    """
    Accion (Box, 2): [fuerza, direccion], ambas en [-1, 1].
        fuerza   se reescala antes de pasarla a la dinamica para que fuerza=0 equivalga a
            una traccion base de `aceleracion_base` (0.5 por defecto): el carro no se queda
            quieto si el agente no hace nada. fuerza=1 pide traccion maxima, fuerza=-1 frena
            a fondo; valores intermedios positivos aumentan la traccion por encima de la base
            y valores negativos la reducen hasta pasar a frenado.
        direccion  angulo objetivo del volante (el real lo sigue con limite de velocidad).

    Observacion (Box, 10): distancia de cada sensor al borde de la pista, normalizada
        por `alcance` (0 = borde pegado, 1 = nada hasta el alcance). Con
        `observar_dinamica=True` (por defecto) se anaden u, v, r normalizadas (3 valores mas); con False solo los rayos.

    Estado interno (`env.unwrapped.estado`): [x, y, psi, u, v, r, delta] + posicion
        en la pista (`info["s"]`, `info["lateral"]`).

    Recompensa por paso:
        peso_velocidad * (u / velocidad_ref) * dt   (u = velocidad longitudinal)
        Con los valores por defecto (peso_velocidad=1), ir justo a velocidad_ref da +1 por
        segundo. No hay penalizacion por chocar: simplemente el episodio termina ahi, asi que
        la recompensa deja de crecer.

    terminated: choque (alguna esquina fuera de la calzada) o vuelta completada.
    truncated:  `max_pasos` alcanzados o sin mejorar el mejor progreso en `tiempo_sin_progreso` s.

    `pista` puede ser un nombre (ver pistas.PISTAS), una Pista, un arreglo (K, 2) de puntos
    de control, o una lista de ellas: en cada reset se elige una al azar.

    `velocidad_inicial` (m/s) es la velocidad longitudinal con la que arranca el carro en cada reset.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        pista="circuito",
        render_mode: str | None = None,
        carro: ParametrosCarro | None = None,
        dt: float = 0.05,
        subpasos: int = 5,
        alcance: float = 60.0,
        max_pasos: int = 3000,
        tiempo_sin_progreso: float = 10.0,
        inicio_aleatorio: bool = False,
        observar_dinamica: bool = True,
        velocidad_inicial: float = 5.0,
        aceleracion_base: float = 0.5,
        velocidad_ref: float = 20.0,
        peso_velocidad: float = 1.0,
    ):
        super().__init__()
        specs = pista if isinstance(pista, (list, tuple)) and not _es_puntos(pista) else [pista]
        self.pistas: list[Pista] = [crear_pista(s) for s in specs]
        self.pista: Pista = self.pistas[0]
        self.carro = carro or ParametrosCarro()
        self.dt, self.subpasos = dt, subpasos
        self.alcance = alcance
        self.max_pasos = max_pasos
        self.max_sin_mejora = int(round(tiempo_sin_progreso / dt))
        self.inicio_aleatorio = inicio_aleatorio
        self.observar_dinamica = observar_dinamica
        self.velocidad_inicial = velocidad_inicial
        self.aceleracion_base = aceleracion_base
        self.velocidad_ref = velocidad_ref
        self.peso_velocidad = peso_velocidad
        self.render_mode = render_mode
        self._visor = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n_obs = len(ANGULOS_SENSORES) + (3 if observar_dinamica else 0)
        bajo = np.concatenate([np.zeros(len(ANGULOS_SENSORES)), -np.ones(n_obs - len(ANGULOS_SENSORES))])
        self.observation_space = spaces.Box(bajo.astype(np.float32), np.ones(n_obs, np.float32), dtype=np.float32)

        self.estado = np.zeros(7)
        self._rayos = np.zeros(len(ANGULOS_SENSORES))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if len(self.pistas) > 1:
            self.pista = self.pistas[int(self.np_random.integers(len(self.pistas)))]
        s0 = float(self.np_random.uniform(0, self.pista.longitud)) if self.inicio_aleatorio else 0.0
        if options and "s0" in options:
            s0 = float(options["s0"])
        pos, rumbo = self.pista.punto_inicial(s0)
        self.estado = np.array([pos[0], pos[1], rumbo, self.velocidad_inicial, 0.0, 0.0, 0.0])
        self._s_prev, _ = self.pista.proyectar(pos)
        self.progreso = 0.0
        self.mejor_progreso = 0.0
        self.pasos = 0
        self.pasos_sin_mejora = 0
        return self._observar(), self._info(False, False)

    def step(self, accion):
        accion = np.clip(np.asarray(accion, dtype=float), -1.0, 1.0)
        fuerza = _reescalar_fuerza(float(accion[0]), self.aceleracion_base)
        h = self.dt / self.subpasos
        for _ in range(self.subpasos):
            self.estado = dinamica.paso(self.estado, fuerza, accion[1], self.carro, h)
        self.pasos += 1

        s, lateral = self.pista.proyectar(self.estado[:2])
        ds = (s - self._s_prev + self.pista.longitud / 2) % self.pista.longitud - self.pista.longitud / 2
        self._s_prev = s
        self.progreso += ds

        choque = self._hay_choque()
        vuelta = (not choque) and self.progreso >= self.pista.longitud

        recompensa = self.peso_velocidad * self.estado[3] / self.velocidad_ref * self.dt

        if self.progreso > self.mejor_progreso + 1e-6:
            self.mejor_progreso, self.pasos_sin_mejora = self.progreso, 0
        else:
            self.pasos_sin_mejora += 1
        terminado = choque or vuelta
        estancado = self.pasos_sin_mejora >= self.max_sin_mejora
        truncado = (not terminado) and (self.pasos >= self.max_pasos or estancado)

        obs = self._observar()
        if self.render_mode == "human":
            self.render()
        return obs, float(recompensa), terminado, truncado, self._info(choque, vuelta, s, lateral)

    def render(self):
        if self.render_mode is None:
            return None
        from .render import Visor

        if self._visor is None or self._visor.pista is not self.pista:
            if self._visor is not None:
                self._visor.cerrar()
            self._visor = Visor(self.pista, self.carro, self.render_mode, self.metadata["render_fps"])
        return self._visor.dibujar(self.estado, self._rayos, ANGULOS_SENSORES)

    def close(self):
        if self._visor is not None:
            self._visor.cerrar()
            self._visor = None

    # ------------------------------------------------------------------ helpers
    def _observar(self) -> np.ndarray:
        self._rayos = self.pista.distancias_rayos(self.estado[:2], self.estado[2], ANGULOS_SENSORES, self.alcance)
        obs = self._rayos / self.alcance
        if self.observar_dinamica:
            u, v, r = self.estado[3:6]
            obs = np.concatenate([obs, np.clip([u / 40.0, v / 10.0, r / 2.0], -1, 1)])
        return obs.astype(np.float32)

    def _hay_choque(self) -> bool:
        for q in dinamica.esquinas(self.estado, self.carro):
            if abs(self.pista.proyectar(q)[1]) > self.pista.semiancho:
                return True
        return False

    def _info(self, choque, vuelta, s=None, lateral=None) -> dict:
        if s is None:
            s, lateral = self.pista.proyectar(self.estado[:2])
        u, v = self.estado[3], self.estado[4]
        return {
            "s": s,
            "lateral": lateral,
            "progreso": self.progreso / self.pista.longitud,
            "velocidad": math.hypot(u, v),
            "derrape": math.atan2(v, max(u, 1.0)),
            "tiempo": self.pasos * self.dt,
            "choque": choque,
            "vuelta_completada": vuelta,
            "tiempo_vuelta": self.pasos * self.dt if vuelta else None,
            "pista": self.pista.nombre,
        }


def _reescalar_fuerza(accion: float, base: float) -> float:
    """Reescala la accion en [-1, 1] a la fuerza real (tambien en [-1, 1]) que recibe la
    dinamica, de modo que accion=0 equivalga a una traccion base de `base` en vez de 0."""
    if accion >= 0.0:
        return base + (1.0 - base) * accion
    return base + (base + 1.0) * accion


def _es_puntos(x) -> bool:
    try:
        a = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        return False
    return a.ndim == 2 and a.shape[1] == 2
