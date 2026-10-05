from gymnasium.envs.registration import register

from .dinamica import ParametrosCarro
from .entorno import ANGULOS_SENSORES, CarroPistaEnv
from .pistas import PISTAS, Pista, crear_pista

register(id="CarroPista-v0", entry_point="carro_rl.entorno:CarroPistaEnv")

__all__ = ["CarroPistaEnv", "ParametrosCarro", "Pista", "PISTAS", "crear_pista", "ANGULOS_SENSORES"]
