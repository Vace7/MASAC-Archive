import numpy as np
from gym import spaces
from factory_sim.model.factory_with_parallel_machines import Machine, Warehouse, Product, Factory
from factory_sim.config import Config


class SingleFactoryEnv:
    def __init__(self, sim_env, factory_name, factory_settings):
        self.sim_env = sim_env

        self.warehouses = {name: Warehouse(self.sim_env, name, **kwargs) for name, kwargs in factory_settings["warehouses"].items()}
        self.machines = {name: Machine(self.sim_env, name, **kwargs) for name, kwargs in factory_settings["machines"].items()}
        self.products = {name: Product(name, **kwargs) for name, kwargs in factory_settings["products"].items()}
        self.factory = Factory(self.sim_env, factory_name, self.machines, self.warehouses, self.products)
        self.factory.run()

        self.n_agents = len(self.machines)
        self.observation_space = [spaces.Box(-1, 1, (13 * len(Config.OBSERVATION_WAREHOUSE[name]) + 2 * len(Config.OBSERVATION_LORRY[name]),), np.float32) for name in self.machines]
        self.action_space = [spaces.Discrete(len(Config.PRODUCTION_RATES))] * len(self.machines)

    def reset(self):
        for name in self.machines:
            self.machines[name].reset()
        for name in self.warehouses:
            self.warehouses[name].reset()
        
    def take_actions(self, actions):
        for i, name in enumerate(self.machines):
            self.machines[name].set_production_rate(actions[i])

    
