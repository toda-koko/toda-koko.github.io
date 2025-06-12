import gymnasium as gym
import three_echelonSupplyChain_env
from gymnasium.envs.registration import register

register(
    id = "ThreeEchelonSupplyChain-v0", 
    entry_point = "three_echelonSupplyChain_env:ThreeEchelonSupplyChain",
)