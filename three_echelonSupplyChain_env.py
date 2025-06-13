import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ThreeEchelonSupplyChain(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_inventory = 100,
        farm_max_inventory = 100,
        factory_max_inventory = 100,
        retail_max_inventory = 100,
        max_order = 50,
        demand_mean = 20,
        demand_std = 5,
        max_steps = 50,
        unit_lostcost = 2.0,
        unit_holdcost = 1.0,
        unit_sale =5.0
    ):
        super().__init__()

        self.max_inventory = max_inventory
        self.farm_max_inventory = farm_max_inventory
        self.factory_max_inventory = factory_max_inventory
        self.retail_max_inventory = retail_max_inventory
        self.max_order = max_order
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.max_steps = max_steps
        self.unit_lostcost = unit_lostcost
        self.unit_holdcost = unit_holdcost
        self.unit_sale = unit_sale
        self.current_step = 0

        self.action_space = spaces.Box(low=0, high=max_order, shape=(2,), dtype=np.int32)

        self.observation_space = spaces.Box(
            low=0,
            high=max_inventory,
            shape=(3,),
            dtype=np.int32
        ) 

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.inventory = {
            "farm": self.farm_max_inventory // 2,
            "factory": self.factory_max_inventory // 2,
            "retail": self.retail_max_inventory // 2,
        } 
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.array([
            self.inventory["farm"],
            self.inventory["factory"],
            self.inventory["retail"]
        ], dtype=np.int32)
    

    def step(self, action):
        action = np.clip(action, 0, self.max_order).astype(int)
        orders = {
            "farm_to_factory": action[0],
            "factory_to_retail": action[1]
        }

        # from farm to factory
        farm_to_factory = min(self.inventory["farm"], orders["farm_to_factory"])
        self.inventory["farm"] -= farm_to_factory
        self.inventory["factory"] += farm_to_factory

        # from factory to retail
        factory_to_retail = min(self.inventory["factory"], orders["factory_to_retail"])
        self.inventory["factory"] -= factory_to_retail
        self.inventory["retail"] += factory_to_retail

        # generate demand
        demand = max(0, int(np.random.normal(self.demand_mean, self.demand_std)))

        # sale（and lost）
        sales = min(self.inventory["retail"], demand)
        self.inventory["retail"] -= sales
        lost_sales = demand - sales 

        # calculate reward（revenue - all cost or penalty）
        revenue = self.unit_sale * sales
        holding_cost = self.unit_holdcost * sum(self.inventory.values())
        stockout_penalty = self.unit_lostcost * lost_sales
        reward = revenue - holding_cost - stockout_penalty

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        print(f"Inventory situation: {self.inventory}")
