import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from carro_rl import CarroPistaEnv, PISTAS
from ejemplos.piloto_heuristico import piloto


@pytest.mark.parametrize("pista", sorted(PISTAS))
def test_check_env(pista):
    check_env(CarroPistaEnv(pista=pista), skip_render_check=True)


def test_observacion_y_estado_interno():
    env = CarroPistaEnv()
    obs, info = env.reset(seed=1)
    assert obs.shape == (7,) and obs.dtype == np.float32
    assert env.unwrapped.estado.shape == (7,)
    assert {"s", "lateral", "progreso", "velocidad"} <= set(info)


def test_choca_a_toda_velocidad_en_linea_recta():
    env = CarroPistaEnv(pista="ovalo")
    env.reset(seed=0)
    for _ in range(600):
        _, r, term, trunc, info = env.step([1.0, 0.0])
        if term or trunc:
            break
    assert term and info["choque"] and r < -5


def test_derrape_en_curva_rapida():
    env = CarroPistaEnv(pista="ovalo")
    env.reset(seed=0, options={"s0": 0})
    derrape = 0.0
    for _ in range(400):
        _, _, term, trunc, info = env.step([1.0, 0.3])
        derrape = max(derrape, abs(info["derrape"]))
        if term or trunc:
            break
    assert derrape > 0.15


def test_vuelta_completa_con_piloto_y_recompensa_positiva():
    env = CarroPistaEnv(pista="ovalo")
    obs, _ = env.reset(seed=0)
    total, fin = 0.0, False
    while not fin:
        obs, r, term, trunc, info = env.step(piloto(obs))
        total += r
        fin = term or trunc
    assert info["vuelta_completada"] and info["tiempo_vuelta"] > 0 and total > 0


def test_truncamiento_por_estar_quieto():
    env = CarroPistaEnv(tiempo_sin_progreso=2.0)
    env.reset(seed=0)
    for _ in range(100):
        _, _, term, trunc, _ = env.step([0.0, 0.0])
        if trunc:
            break
    assert trunc and not term


def test_varias_pistas_y_inicio_aleatorio():
    env = CarroPistaEnv(pista=["ovalo", "circuito", "trebol"], inicio_aleatorio=True)
    vistas = {env.reset(seed=i)[1]["pista"] for i in range(30)}
    assert len(vistas) == 3


def test_observacion_extendida():
    env = CarroPistaEnv(observacion_extendida=True)
    assert env.reset(seed=0)[0].shape == (10,)
