"""Visualizacion con matplotlib (import perezoso: el entorno funciona sin ella)."""

from __future__ import annotations

import numpy as np

from . import dinamica


class Visor:
    def __init__(self, pista, carro, modo: str, fps: int):
        import matplotlib

        if modo == "rgb_array":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        self.plt, self.pista, self.carro, self.modo, self.fps = plt, pista, carro, modo, fps
        self.fig, self.ax = plt.subplots(figsize=(7, 7), dpi=90)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.patch.set_facecolor("#6f7f8c")
        self.ax.set_facecolor("#6f7f8c")
        izq, der = pista.borde_izq, pista.borde_der
        calzada = np.vstack([izq, izq[:1], der[:1], der[::-1]])
        self.ax.add_patch(Polygon(calzada, closed=True, fc="#3b4650", ec="none"))
        for borde in (izq, der):
            cerrado = np.vstack([borde, borde[:1]])
            self.ax.plot(cerrado[:, 0], cerrado[:, 1], color="white", lw=2)
        self.ax.plot(*np.stack([izq[0], der[0]]).T, color="white", lw=3)  # linea de salida
        self.ax.autoscale_view()
        self.cuerpo = Polygon(np.zeros((4, 2)), closed=True, fc="#2fbf5b", ec="black", zorder=5)
        self.ax.add_patch(self.cuerpo)
        self.lineas = [self.ax.plot([], [], color="white", lw=0.8, alpha=0.9, zorder=4)[0] for _ in range(7)]
        self.texto = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top", color="white", family="monospace")
        if modo == "human":
            plt.ion()
            self.fig.show()

    def dibujar(self, estado, rayos, angulos):
        self.cuerpo.set_xy(dinamica.esquinas(estado, self.carro))
        for ln, d, a in zip(self.lineas, rayos, angulos):
            ang = estado[2] + a
            ln.set_data([estado[0], estado[0] + d * np.cos(ang)], [estado[1], estado[1] + d * np.sin(ang)])
        self.texto.set_text(f"v = {np.hypot(estado[3], estado[4]) * 3.6:5.1f} km/h")
        if self.modo == "human":
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(1.0 / self.fps)
            return None
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def cerrar(self):
        self.plt.close(self.fig)
