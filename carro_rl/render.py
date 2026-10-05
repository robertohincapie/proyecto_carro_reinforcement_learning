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
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.mini = self.fig.add_axes([0.02, 0.02, 0.27, 0.27], zorder=10)
        fondo = "#6f7f8c"
        self.fig.patch.set_facecolor(fondo)
        izq, der = pista.borde_izq, pista.borde_der
        calzada = np.vstack([izq, izq[:1], der[:1], der[::-1]])
        for ax in (self.ax, self.mini):  # misma pista en la vista principal y en el minimapa
            ax.set_aspect("equal")
            ax.set_facecolor(fondo)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.add_patch(Polygon(calzada, closed=True, fc="#3b4650", ec="none"))
            for borde in (izq, der):
                cerrado = np.vstack([borde, borde[:1]])
                ax.plot(cerrado[:, 0], cerrado[:, 1], color="white", lw=2 if ax is self.ax else 1)
            ax.plot(*np.stack([izq[0], der[0]]).T, color="white", lw=3 if ax is self.ax else 1.5)  # salida
        # Vista principal: recuadro centrado en el carro; el minimapa muestra la pista completa.
        self.mitad = 30.0
        self.ax.axis("off")
        lim = np.vstack([izq, der])
        margen = 15.0
        self.mini.set_xlim(lim[:, 0].min() - margen, lim[:, 0].max() + margen)
        self.mini.set_ylim(lim[:, 1].min() - margen, lim[:, 1].max() + margen)
        for sp in self.mini.spines.values():
            sp.set_edgecolor("white")
            sp.set_linewidth(1.5)
        self.punto = self.mini.plot([], [], "o", color="#ff4d4d", ms=6, zorder=6)[0]
        self.cuerpo = Polygon(np.zeros((4, 2)), closed=True, fc="#2fbf5b", ec="black", zorder=5)
        self.ax.add_patch(self.cuerpo)
        self.lineas = [self.ax.plot([], [], color="white", lw=0.8, alpha=0.9, zorder=4)[0] for _ in range(7)]
        self.texto = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top", color="white", family="monospace")
        if modo == "human":
            plt.ion()
            self.fig.show()

    def dibujar(self, estado, rayos, angulos):
        x, y = estado[0], estado[1]
        self.ax.set_xlim(x - self.mitad, x + self.mitad)
        self.ax.set_ylim(y - self.mitad, y + self.mitad)
        self.punto.set_data([x], [y])
        self.cuerpo.set_xy(dinamica.esquinas(estado, self.carro))
        for ln, d, a in zip(self.lineas, rayos, angulos):
            ang = estado[2] + a
            ln.set_data([x, x + d * np.cos(ang)], [y, y + d * np.sin(ang)])
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
