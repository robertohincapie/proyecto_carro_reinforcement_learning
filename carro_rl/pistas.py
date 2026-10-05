"""Pistas cerradas definidas por una linea central (spline periodico) y un ancho."""

from __future__ import annotations

import numpy as np


def _catmull_rom_cerrado(puntos: np.ndarray, por_tramo: int = 60) -> np.ndarray:
    n = len(puntos)
    t = np.linspace(0.0, 1.0, por_tramo, endpoint=False)[:, None]
    tramos = []
    for i in range(n):
        p0, p1, p2, p3 = (puntos[(i + k) % n] for k in (-1, 0, 1, 2))
        tramos.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    return np.vstack(tramos)


def _rotar(v: np.ndarray, ang: np.ndarray) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)


class Pista:
    """Circuito cerrado.

    Parameters
    ----------
    puntos : (K, 2) puntos de control (metros) por los que pasa la linea central.
    ancho : ancho total de la calzada en metros.
    muestras : numero de puntos equiespaciados de la linea central.
    """

    def __init__(self, puntos, ancho: float = 12.0, muestras: int = 900, nombre: str = "personalizada"):
        puntos = np.asarray(puntos, dtype=float)
        if puntos.ndim != 2 or puntos.shape[1] != 2 or len(puntos) < 4:
            raise ValueError("puntos debe tener forma (K, 2) con K >= 4")
        self.nombre = nombre
        self.semiancho = ancho / 2.0

        fino = _catmull_rom_cerrado(puntos)
        cerrado = np.vstack([fino, fino[:1]])
        seg = np.linalg.norm(np.diff(cerrado, axis=0), axis=1)
        acum = np.concatenate([[0.0], np.cumsum(seg)])
        self.longitud = float(acum[-1])
        self.n = muestras
        self.ds = self.longitud / muestras
        objetivo = np.arange(muestras) * self.ds
        self.centro = np.stack(
            [np.interp(objetivo, acum, cerrado[:, 0]), np.interp(objetivo, acum, cerrado[:, 1])], axis=1
        )
        self.s = objetivo

        # Tangentes, normales (izquierda) y curvatura
        sig = np.roll(self.centro, -1, axis=0)
        ant = np.roll(self.centro, 1, axis=0)
        tang = sig - ant
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        self.tang = tang
        self.normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        rumbo = np.arctan2(tang[:, 1], tang[:, 0])
        dr = np.angle(np.exp(1j * (np.roll(rumbo, -1) - rumbo)))
        self.curvatura = dr / self.ds
        radio_min = 1.0 / max(np.abs(self.curvatura).max(), 1e-9)
        if radio_min < self.semiancho * 1.05:
            raise ValueError(
                f"Pista '{nombre}': radio de curvatura minimo {radio_min:.1f} m < semiancho {self.semiancho:.1f} m; "
                "los bordes se cruzarian. Suaviza la pista o reduce el ancho."
            )
        self.radio_min = radio_min

        self.borde_izq = self.centro + self.normal * self.semiancho
        self.borde_der = self.centro - self.normal * self.semiancho
        a = np.vstack([self.borde_izq, self.borde_der])
        b = np.vstack([np.roll(self.borde_izq, -1, axis=0), np.roll(self.borde_der, -1, axis=0)])
        self._seg_a, self._seg_e = a, b - a

    # --------------------------------------------------------------- geometria
    def punto_inicial(self, s: float = 0.0):
        """Posicion y rumbo (rad) sobre la linea central en la coordenada s."""
        i = int(round((s % self.longitud) / self.ds)) % self.n
        return self.centro[i].copy(), float(np.arctan2(self.tang[i, 1], self.tang[i, 0]))

    def proyectar(self, p: np.ndarray):
        """Proyecta p sobre la linea central. Devuelve (s, desplazamiento_lateral con signo (+ izquierda))."""
        d2 = np.sum((self.centro - p) ** 2, axis=1)
        i = int(np.argmin(d2))
        mejor = None
        for j in (i - 1, i):  # tramos (i-1 -> i) e (i -> i+1)
            a = self.centro[j % self.n]
            e = self.centro[(j + 1) % self.n] - a
            t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0.0, 1.0)
            q = a + t * e
            dist = np.linalg.norm(p - q)
            if mejor is None or dist < mejor[0]:
                signo = np.sign(e[0] * (p[1] - q[1]) - e[1] * (p[0] - q[0]))
                mejor = (dist, ((j % self.n) + t) * self.ds, signo * dist)
        return mejor[1], mejor[2]

    def distancias_rayos(self, pos: np.ndarray, rumbo: float, angulos: np.ndarray, alcance: float) -> np.ndarray:
        """Distancia de pos al borde mas cercano a lo largo de cada rayo (saturada en alcance)."""
        ang = rumbo + angulos
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (R, 2)
        ao = self._seg_a - pos  # (M, 2)
        e = self._seg_e  # (M, 2)
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # (R, M)
        cruz_ae = ao[:, 0] * e[:, 1] - ao[:, 1] * e[:, 0]  # (M,)
        cruz_ad = ao[None, :, 0] * d[:, None, 1] - ao[None, :, 1] * d[:, None, 0]  # (R, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cruz_ae[None, :] / denom
            u = cruz_ad / denom
        valido = (np.abs(denom) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
        t = np.where(valido, t, np.inf)
        return np.minimum(t.min(axis=1), alcance)


# ---------------------------------------------------------------- pistas listas
def _ovalo() -> Pista:
    th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    return Pista(np.stack([170 * np.cos(th), 90 * np.sin(th)], axis=1), ancho=12, nombre="ovalo")


def _circuito() -> Pista:
    th = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    r = 150 * (1 + 0.28 * np.sin(2 * th + 0.6) + 0.18 * np.sin(3 * th + 1.9) + 0.07 * np.sin(5 * th))
    return Pista(np.stack([r * np.cos(th), r * np.sin(th)], axis=1), ancho=12, nombre="circuito")


def _trebol() -> Pista:
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    r = 160 * (1 + 0.32 * np.cos(3 * th))
    return Pista(np.stack([r * np.cos(th), r * np.sin(th)], axis=1), ancho=12, nombre="trebol")


PISTAS = {"ovalo": _ovalo, "circuito": _circuito, "trebol": _trebol}


def crear_pista(spec) -> Pista:
    """Acepta un nombre de PISTAS, una Pista o un arreglo (K, 2) de puntos de control."""
    if isinstance(spec, Pista):
        return spec
    if isinstance(spec, str):
        if spec not in PISTAS:
            raise ValueError(f"Pista desconocida '{spec}'. Disponibles: {sorted(PISTAS)}")
        return PISTAS[spec]()
    return Pista(spec)
