import simpy
import gym
from factory_sim.model.transport_old import Logistic_center, Lorry
from factory_sim.environment.single_factory_env import SingleFactoryEnv
from factory_sim.config import Config
import numpy as np
import pygame

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
OUTLINE_WIDTH = 2
ARROW_RAD = 5
PLUS_SIGN_SIZE = 21
BLACK = (0,0,0)
GREEN = (0,255,0)
YELLOW = (255, 204, 0)
WHITE = (255,255,255)
RED = (255,0,0)
GREY = (128,128,128)

FACTORY_HEIGHT = 250
FACTORY_WIDTH = 400

MACHINE_HEIGHT = 63
MACHINE_WIDTH = 80

CONTAINER_HEIGHT = MACHINE_HEIGHT
CONTAINER_WIDTH = 20

LORRY_WIDTH = 20
LORRY_HEIGHT = 10

FACTORY_FONT_SIZE = 16
MACHINE_FONT_SIZE = 12
CONTAINER_FONT_SIZE = 8

RIGHT_X = SCREEN_WIDTH - FACTORY_WIDTH
BOTTOM_Y = SCREEN_HEIGHT - FACTORY_HEIGHT

class Meter:
	def __init__(self):
		self.data = []

	def reset(self):
		self.data.clear()

	def add(self, v):
		self.data.append(v)

	def diff(self, pos1, pos2):
		if len(self.data) < 2:
			return 0

		return self.data[pos1] - self.data[pos2]


class MultiFactoryEnv(gym.Env):
    # --------------------------------------Env Init--------------------------------------
    def __init__(self, render_mode=None, render_fps=1):
        super().__init__()
        self.sim_env = simpy.Environment()
        self.simulate_step_time = Config.DECISION_INTERVAL
        self.simulate_until = 0
        self.env_step_cont = 0
        self.sim_time_counter = 0
        self.n_steps = Config.N_STEPS

        self.factory_1 = SingleFactoryEnv(self.sim_env, "factory_1", Config.FACTORY_1)
        self.factory_2 = SingleFactoryEnv(self.sim_env, "factory_2", Config.FACTORY_2)
        self.factory_3 = SingleFactoryEnv(self.sim_env, "factory_3", Config.FACTORY_3)
        self.factory_4 = SingleFactoryEnv(self.sim_env, "factory_4", Config.FACTORY_4)
        self.factories = {"factory_1": self.factory_1, "factory_2": self.factory_2, "factory_3": self.factory_3, "factory_4": self.factory_4}

        self.lorries = {name: Lorry(self.sim_env, name, **kwargs) for name, kwargs in Config.LORRIES.items()}
        self.logistic_center = Logistic_center(self.sim_env, "logistic_center", self.factories, self.lorries)

        self.n_agents = sum([self.factories[name].n_agents for name in self.factories])
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

        self.reward_meters = {tar_wh: Meter() for tar_wh in Config.REWARD_INFO["target_warehouses"]}
        self.total_revenue = 0
        self.total_energy_cost = 0
        self.total_storage_cost = 0

        self.container_level_history = []
        self.energy_cost_history = []
        self.iot_sensors = {}
        for i, factory in enumerate([Config.FACTORY_1, Config.FACTORY_2, Config.FACTORY_3, Config.FACTORY_4]):
            self.iot_sensors[f"factory_{i+1}"] = {
                "warehouses": {key: 0 for key, _ in factory["warehouses"].items()},
                "machines": {key: 0 for key, _ in factory["machines"].items()}
            }
            del self.iot_sensors[f"factory_{i+1}"]["warehouses"][f"in_warehouse_{i+1}"]
        self.iot_sensors["lorries"] = {key: 0 for key, _ in Config.LORRIES.items()}

        # --------------------------------------Rendering Init--------------------------------------
        self.render_mode = render_mode
        self.render_fps = render_fps

        if render_mode == "human":
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )
            self.clock = pygame.time.Clock()
            pygame.init()
            pygame.display.init()
            self.factory_1_obj = pygame.Rect((1,1,FACTORY_WIDTH,FACTORY_HEIGHT))
            self.factory_2_obj = pygame.Rect((RIGHT_X,1,FACTORY_WIDTH,FACTORY_HEIGHT))
            self.factory_3_obj = pygame.Rect((RIGHT_X,BOTTOM_Y,FACTORY_WIDTH,FACTORY_HEIGHT))
            self.factory_4_obj = pygame.Rect((1,BOTTOM_Y,FACTORY_WIDTH,FACTORY_HEIGHT))

            self.machine_1 = pygame.Rect((0.5*MACHINE_WIDTH,94,MACHINE_WIDTH,MACHINE_HEIGHT))
            self.machine_2 = pygame.Rect((RIGHT_X + 3.5*MACHINE_WIDTH,4,MACHINE_WIDTH,MACHINE_HEIGHT))
            self.machine_3 = pygame.Rect((RIGHT_X + MACHINE_WIDTH+20,4,MACHINE_WIDTH,MACHINE_HEIGHT))
            self.machine_4 = pygame.Rect((RIGHT_X + 2.35*MACHINE_WIDTH, BOTTOM_Y + 2*MACHINE_WIDTH,MACHINE_WIDTH,MACHINE_HEIGHT))
            self.machine_5 = pygame.Rect((RIGHT_X + MACHINE_WIDTH+20,BOTTOM_Y + MACHINE_WIDTH,MACHINE_WIDTH,MACHINE_HEIGHT))
            self.machine_6 = pygame.Rect((RIGHT_X + 3.5*MACHINE_WIDTH,BOTTOM_Y + MACHINE_WIDTH,MACHINE_WIDTH,MACHINE_HEIGHT))
            self.machine_7 = pygame.Rect((0.5*MACHINE_WIDTH,BOTTOM_Y + MACHINE_WIDTH,MACHINE_WIDTH,MACHINE_HEIGHT))
            self.machine_8 = pygame.Rect((3*MACHINE_WIDTH,BOTTOM_Y + MACHINE_WIDTH,MACHINE_WIDTH,MACHINE_HEIGHT))

            self.container_1 = pygame.Rect((self.machine_1.right + MACHINE_WIDTH, self.machine_1.top,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_2 = pygame.Rect((self.machine_3.left-0.5*MACHINE_WIDTH,self.machine_3.top,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_3 = pygame.Rect((self.machine_2.centerx,self.machine_2.bottom + MACHINE_HEIGHT,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_4 = pygame.Rect((self.machine_3.right + 0.5*MACHINE_WIDTH,self.machine_3.top,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_5 = pygame.Rect((self.machine_3.centerx,self.machine_3.bottom + MACHINE_HEIGHT,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_6 = pygame.Rect((self.machine_5.centerx,self.machine_5.top-MACHINE_HEIGHT-10,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_7 = pygame.Rect((self.machine_4.centerx-1.5*CONTAINER_WIDTH,self.machine_5.top,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_8 = pygame.Rect((self.machine_4.centerx+CONTAINER_WIDTH,self.machine_5.top,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_9 = pygame.Rect((self.machine_6.centerx,self.machine_6.top-MACHINE_HEIGHT-10,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_10 = pygame.Rect((self.machine_5.left-0.5*MACHINE_WIDTH,self.machine_5.top,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_11 = pygame.Rect((self.machine_8.right + 0.5*MACHINE_WIDTH,self.machine_8.top,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_12 = pygame.Rect((self.machine_7.right + 0.5*MACHINE_WIDTH, self.machine_7.top,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_13 = pygame.Rect((self.machine_6.centerx,BOTTOM_Y + 2*MACHINE_WIDTH,CONTAINER_WIDTH,CONTAINER_HEIGHT))
            self.container_14 = pygame.Rect((self.machine_8.centerx,BOTTOM_Y + 2*MACHINE_WIDTH,CONTAINER_WIDTH,CONTAINER_HEIGHT))

            self.lorry_1 = pygame.Rect((self.factory_1_obj.right,self.factory_1_obj.centery,LORRY_WIDTH,LORRY_HEIGHT))
            self.lorry_2 = pygame.Rect((self.container_3.left,self.factory_2_obj.bottom,LORRY_WIDTH,LORRY_HEIGHT))
            self.lorry_3 = pygame.Rect((self.container_5.left,self.factory_2_obj.bottom,LORRY_WIDTH,LORRY_HEIGHT))
            self.lorry_4 = pygame.Rect((self.factory_3_obj.left - LORRY_WIDTH,self.container_10.centery,LORRY_WIDTH,LORRY_HEIGHT))
            self.resource_map = {
                'factory_1' : 'factory_1_obj',
                'factory_2' : 'factory_2_obj',
                'factory_3' : 'factory_3_obj',
                'factory_4' : 'factory_4_obj',
                'machine_1' : {'obj' : 'machine_1', 'name' : 'M1'},
                'machine_2' : {'obj' : 'machine_2', 'name' : 'M2'},
                'machine_1-2' : {'obj' : 'machine_3', 'name' : 'M3'},
                'machine_3' : {'obj' : 'machine_4', 'name' : 'M4'},
                'machine_1-2-3' : {'obj' : 'machine_6', 'name' : 'M6'},
                'machine_2-3' : {'obj' : 'machine_5', 'name' : 'M5'},
                'machine_4' : {'obj' : 'machine_7', 'name' : 'M7'},
                'machine_2-3-4' : {'obj' : 'machine_8', 'name' : 'M8'},
                'export_warehouse_1' : {'obj' : 'container_1', 'name' : 'C1'},
                'import_warehouse_1' : {'obj' : 'container_2', 'name' : 'C2'},
                'out_warehouse_2' : {'obj' : 'container_4', 'name' : 'C4'},
                'export_warehouse_2' : {'obj' : 'container_3', 'name' : 'C3'},
                'export_warehouse_1-2' : {'obj' : 'container_5', 'name' : 'C5'},
                'out_warehouse_3a' : {'obj' : 'container_7', 'name' : 'C7'},
                'out_warehouse_3b' : {'obj' : 'container_8', 'name' : 'C8'},
                'import_warehouse_1-2' : {'obj' : 'container_6', 'name' : 'C6'},
                'import_warehouse_2' : {'obj' : 'container_9', 'name' : 'C9'},
                'export_warehouse_2-3' : {'obj' : 'container_10', 'name' : 'C10'},
                'out_warehouse_1-2-3' : {'obj' : 'container_13', 'name' : 'C13'},
                'out_warehouse_4' : {'obj' : 'container_12', 'name' : 'C12'},
                'import_warehouse_2-3' : {'obj' : 'container_11', 'name' : 'C11'},
                'out_warehouse_2-3-4' : {'obj' : 'container_14', 'name' : 'C14'},
                'lorry_1' : {'obj' : 'lorry_1', 'dest' : (self.factory_2_obj.left - LORRY_WIDTH, self.container_2.centery)},
                'lorry_2' : {'obj' : 'lorry_2', 'dest' : (self.container_3.left, self.factory_3_obj.top - LORRY_HEIGHT)},
                'lorry_1-2' : {'obj' : 'lorry_3', 'dest' : (self.container_5.left, self.factory_3_obj.top - LORRY_HEIGHT)},
                'lorry_2-3' : {'obj' : 'lorry_4', 'dest' : (self.factory_4_obj.right, self.container_11.centery)}
            }

    # --------------------------------------Main functions--------------------------------------
    def reset(self):
        for name in self.factories:
            self.factories[name].reset()
        for name in self.lorries:
            self.lorries[name].reset()
        for name in self.reward_meters:
            self.reward_meters[name].reset()

        self.env_step_cont = 0
        self.container_level_history = []
        self.energy_cost_history = []
        self.record_container_level()
        if self.render_mode == "human":
            self.render()
        obs = self.get_obs()
        return obs

    def step(self, actions):
        idx = 0
        # Set machine production rates
        for f_name in self.factories:
            minibatch_actions = actions[idx:idx+self.factories[f_name].n_agents]
            self.factories[f_name].take_actions(minibatch_actions)
            idx += self.factories[f_name].n_agents

        # Simpy functions that drive the entire simulation
        self.simulate_until += self.simulate_step_time
        while self.sim_env.peek() < self.simulate_until:
            self.sim_env.step()
            if self.render_mode == "human":
                self.render()
            if self.sim_env.peek() > self.sim_time_counter:
                self.record_container_level()
                self.sim_time_counter += 1
        self.env_step_cont += 1

        obs = self.get_obs()
        rew, revenue = self.get_reward()
        self.total_revenue += sum(revenue)
        storage_cost = self.get_storage_cost()
        self.total_storage_cost += sum(storage_cost)
        energy_cost = self.energy_cost_history[-1]
        self.total_energy_cost += sum(energy_cost)

        done = [(self.env_step_cont >= self.n_steps)] * self.n_agents

        return obs, rew, done, [revenue, energy_cost, storage_cost]

    # -----------------------------------------------------------------Helper functions-----------------------------------------------------------------
    # Records current container levels into self.container_level_history
    # Container_level_history is used to calculate revenue and storage costs
    def record_container_level(self):
        container_level = []
        for f_name in self.factories:
            for w_name in Config.WAREHOUSES_UNDER_MONITOR[f_name]:
                container_level.append(self.factories[f_name].warehouses[w_name].storage.level)
        self.container_level_history.append(container_level)
    
    # --------------------------------------Observation functions--------------------------------------
    # Obtain observations for Step and Reset.
    def get_obs(self):
        # obs = []
        # storage_status = self.gather_storage_status()
        # for m_name in Config.OBSERVATION_WAREHOUSE:
        #     obs.append(np.concatenate((self.get_machine_storage_obs(m_name, storage_status), self.get_lorry_status_obs(m_name))))
        
        obs = np.concatenate(np.array(self.gather_storage_status(), np.float32) / Config.STORAGE_CAPACITY)
        lorry_obs = np.concatenate(np.array([lorry_obj.get_operation_status() for lorry_obj in self.lorries.values()]))
        obs = np.concatenate([obs, lorry_obs])
        return obs
    
    # Used to get container levels and past 6 input and output history
    # Returns information as one list
    def gather_storage_status(self):
        storage_status = []
        for f_name in self.factories:
            for w_name in Config.WAREHOUSES_UNDER_MONITOR[f_name]:
                storage_level = self.factories[f_name].warehouses[w_name].get_storage_status()
                storage_status.append(storage_level)
        return storage_status
    
    # Takes storage status that each machine can observe 
    def get_machine_storage_obs(self, m_name, storage_status):
        obs = []
        for w_name in Config.OBSERVATION_WAREHOUSE[m_name]:
            obs += storage_status[Config.WAREHOUSES_LIST.index(w_name)]
        return np.array(obs, np.float32) / Config.STORAGE_CAPACITY

    # Takes lorry status that each machine can observe 
    def get_lorry_status_obs(self, m_name):
        obs = []
        for l_name in Config.OBSERVATION_LORRY[m_name]:
            obs += self.lorries[l_name].get_operation_status()
        return np.array(obs, np.float32)
    
    # --------------------------------------Reward functions--------------------------------------
    # Reward is a measure of revenue, energy cost and storage cost
    def get_reward(self):
        revenue = self.get_revenue()
        rew = sum(revenue) - 0.01 * sum(self.get_storage_cost()) - 2 * sum(self.get_energy_cost())
        return [rew * 0.01] * self.n_agents, revenue
    
    # Revenue calculates how much selling material was produces in the last step and subtracts the raw material cost from each machine
    def get_revenue(self):
        rev = [0,0,0,0]
        for i, tar_wh in enumerate(Config.REWARD_INFO["target_warehouses"]):
            self.reward_meters[tar_wh].add(self.container_level_history[-1][Config.WAREHOUSES_LIST.index(tar_wh)])
            rev[i+2] += self.reward_meters[tar_wh].diff(-1,-2) * Config.REWARD_INFO["prices"][i]
        for f_name in self.factories:
            for m_name in self.factories[f_name].machines:
                rev[int(f_name[-1])-1] -= self.factories[f_name].machines[m_name].get_material_total_price()
        return rev

    # Calculates the storage cost of the materials produced in the last step
    def get_storage_cost(self):
        s_cost = []
        costs = np.array(self.container_level_history[-Config.DECISION_INTERVAL:])*np.array(Config.VOLUME_COEFF)
        s_cost.append(np.sum(costs[:1]))
        s_cost.append(np.sum(costs[1:5]))
        s_cost.append(np.sum(costs[5:11]))
        s_cost.append(np.sum(costs[11:14]))
        return s_cost

    # Energy costs are calculated by machine and appended to energy cost history, total energy cost at last step is summed and returned
    def get_energy_cost(self):
        energy_cost_per_machine = []
        energy_cost = []
        for f_name in self.factories:
            for m_name in self.factories[f_name].machines:
                energy_cost_per_machine.append(self.factories[f_name].machines[m_name].get_energy_cost())
        costs = np.multiply(energy_cost_per_machine, list(Config.ENERGY_COST_COEFF.values()))
        energy_cost.append(np.sum(costs[:1]))
        energy_cost.append(np.sum(costs[1:5]))
        energy_cost.append(np.sum(costs[5:11]))
        energy_cost.append(np.sum(costs[11:14]))
        self.energy_cost_history.append(energy_cost)
        return energy_cost
    
    # --------------------------------------Rendering functions--------------------------------------

    def get_lorry_pos(self, tup1, tup2, progress):
        x = progress*(tup2[0] - tup1[0]) + tup1[0]
        y = progress*(tup2[1] - tup1[1]) + tup1[1]
        return (x,y,LORRY_WIDTH,LORRY_HEIGHT)
    
    def arrow(self, lcolor, tricolor, start, end, trirad, thickness=2):
        pygame.draw.line(self.screen, lcolor, start, end, thickness)
        rotation = (np.arctan2(start[1] - end[1], end[0] - start[0])) + np.pi/2
        rad = np.pi/180
        pygame.draw.polygon(self.screen, tricolor, ((end[0] + trirad * np.sin(rotation),
                                            end[1] + trirad * np.cos(rotation)),
                                        (end[0] + trirad * np.sin(rotation - 120*rad),
                                            end[1] + trirad * np.cos(rotation - 120*rad)),
                                        (end[0] + trirad * np.sin(rotation + 120*rad),
                                            end[1] + trirad * np.cos(rotation + 120*rad))))
        
    def draw_plus(self, x, y, width, height, thickness=0):
        # Horizontal line
        pygame.draw.rect(self.screen, GREY, (x - width/2, y + (height / 3) - height/2, width, height / 3), thickness)
        # Vertical line
        pygame.draw.rect(self.screen, GREY, (x + (width / 3) - width/2, y - height/2, width / 3, height), thickness)

    def render(self):
        if self.render_mode == "human":
            self.get_sensor_readings()
            self.screen.fill((255,255,255))
            self.screen.blit(pygame.font.Font(pygame.font.get_default_font(), FACTORY_FONT_SIZE).render(\
                f"Revenue: {'${:,.2f}'.format(self.total_revenue)} | Energy Costs: {'${:,.2f}'.format(2*self.total_energy_cost)} | Storage Costs: {'${:,.2f}'.format(0.01*self.total_storage_cost)}",\
                    False,BLACK),(5, 300))
            for f in self.iot_sensors:
                if f!='lorries':
                    f_obj = getattr(self,self.resource_map[f])
                    pygame.draw.rect(self.screen, BLACK, f_obj, OUTLINE_WIDTH)
                    self.screen.blit(pygame.font.Font(pygame.font.get_default_font(), FACTORY_FONT_SIZE).render(f,False,BLACK),(f_obj.left + 5, f_obj.bottom - 18))
                    for m in self.iot_sensors[f]['machines']:
                        obj = getattr(self,self.resource_map[m]['obj'])
                        name = self.resource_map[m]['name']
                        val = self.iot_sensors[f]['machines'][m]
                        pygame.draw.rect(self.screen, BLACK, obj, OUTLINE_WIDTH)
                        pygame.draw.rect(self.screen, BLACK, pygame.Rect((obj.left, obj.top + (1 - min(val,10)/10)*MACHINE_HEIGHT, MACHINE_WIDTH,(min(val,10)/10)*MACHINE_HEIGHT)))
                        self.screen.blit(pygame.font.Font(pygame.font.get_default_font(), MACHINE_FONT_SIZE).render(name,False,BLACK),obj.bottomleft)
                    for c in self.iot_sensors[f]['warehouses']:
                        obj = getattr(self,self.resource_map[c]['obj'])
                        name = self.resource_map[c]['name']
                        val = self.iot_sensors[f]['warehouses'][c]
                        pygame.draw.rect(self.screen, BLACK, obj, OUTLINE_WIDTH)
                        if c in Config.REWARD_INFO["target_warehouses"]:
                            pygame.draw.rect(self.screen, GREEN, pygame.Rect((obj.left, obj.top + (1 - min(val,100)/100)*CONTAINER_HEIGHT, CONTAINER_WIDTH,(min(val,100)/100)*CONTAINER_HEIGHT)))
                        else:
                            pygame.draw.rect(self.screen, YELLOW, pygame.Rect((obj.left, obj.top + (1 - min(val,100)/100)*CONTAINER_HEIGHT, CONTAINER_WIDTH,(min(val,100)/100)*CONTAINER_HEIGHT)))
                        self.screen.blit(pygame.font.Font(pygame.font.get_default_font(), CONTAINER_FONT_SIZE).render(name,False,BLACK),obj.bottomleft)
                else:
                    for l in self.iot_sensors[f]:
                        obj = getattr(self,self.resource_map[l]['obj'])
                        dest = self.resource_map[l]['dest']
                        val = self.iot_sensors[f][l][1]
                        pygame.draw.rect(self.screen, BLACK, pygame.Rect(self.get_lorry_pos(obj.topleft, dest, val)))
            
            # lines
            self.draw_plus(self.machine_3.centerx,self.machine_3.centery,PLUS_SIGN_SIZE,PLUS_SIGN_SIZE)
            self.draw_plus(self.machine_5.centerx,self.machine_5.centery,PLUS_SIGN_SIZE,PLUS_SIGN_SIZE)
            self.draw_plus(self.machine_6.centerx,self.machine_6.centery,PLUS_SIGN_SIZE,PLUS_SIGN_SIZE)
            self.draw_plus(self.machine_8.centerx,self.machine_8.centery,PLUS_SIGN_SIZE,PLUS_SIGN_SIZE)
            self.arrow(BLACK,BLACK,(self.machine_1.right,self.machine_1.centery),(self.container_1.left,self.container_1.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.container_2.right,self.container_2.centery),(self.machine_3.left,self.machine_3.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.container_4.left,self.container_4.centery),(self.machine_3.right,self.machine_3.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.machine_2.left,self.machine_2.centery),(self.container_4.right,self.container_4.centery),ARROW_RAD,OUTLINE_WIDTH)

            self.arrow(BLACK,BLACK,(self.machine_3.centerx,self.machine_3.bottom),self.container_5.topleft,ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.machine_2.centerx,self.machine_2.bottom),self.container_3.topleft,ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,self.container_6.bottomleft,(self.machine_5.centerx,self.machine_5.top),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,self.container_9.bottomleft,(self.machine_6.centerx,self.machine_6.top),ARROW_RAD,OUTLINE_WIDTH)

            self.arrow(BLACK,BLACK,(self.container_7.left,self.container_7.centery),(self.machine_5.right,self.machine_5.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.container_8.right,self.container_8.centery),(self.machine_6.left,self.machine_6.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,self.machine_4.topleft,self.container_7.bottomleft,ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,self.machine_4.topright,self.container_8.bottomright,ARROW_RAD,OUTLINE_WIDTH)

            self.arrow(BLACK,BLACK,(self.machine_6.centerx,self.machine_6.bottom),self.container_13.topleft,ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.machine_5.left,self.machine_5.centery),(self.container_10.right,self.container_10.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.machine_8.centerx,self.machine_8.bottom),self.container_14.topleft,ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.container_11.left,self.container_11.centery),(self.machine_8.right,self.machine_8.centery),ARROW_RAD,OUTLINE_WIDTH)

            self.arrow(BLACK,BLACK,(self.container_12.right,self.container_12.centery),(self.machine_8.left,self.machine_8.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.machine_7.right,self.machine_7.centery),(self.container_12.left,self.container_12.centery),ARROW_RAD,OUTLINE_WIDTH)

            self.arrow(BLACK,BLACK,(self.container_1.right,self.container_1.centery),(self.factory_1_obj.right,self.container_1.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.factory_2_obj.left,self.container_2.centery),(self.container_2.left,self.container_2.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,self.container_5.bottomright,self.lorry_3.topright,ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,self.container_3.bottomright,self.lorry_2.topright,ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.container_6.left,self.factory_3_obj.top),(self.container_6.left,self.container_6.top),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.container_9.left,self.factory_3_obj.top),(self.container_9.left,self.container_9.top),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.container_10.left,self.container_10.centery),(self.factory_3_obj.left,self.container_10.centery),ARROW_RAD,OUTLINE_WIDTH)
            self.arrow(BLACK,BLACK,(self.factory_4_obj.right,self.container_11.centery),(self.container_11.right,self.container_11.centery),ARROW_RAD,OUTLINE_WIDTH)

            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()

    # --------------------------------------Contextual Gym Functions--------------------------------------
    def get_observation_space(self):
        obs_space = []
        for name in self.factories:
            obs_space += self.factories[name].observation_space
        return obs_space

    def get_action_space(self):
        act_space = []
        for name in self.factories:
            act_space += self.factories[name].action_space
        return act_space

    # --------------------------------------IoT functions--------------------------------------
    # Function to get IoT sensor readings for Digital Twin
    def get_sensor_readings(self):
        for f_name in self.factories:
            for w_name in Config.WAREHOUSES_UNDER_MONITOR[f_name]:
                if w_name in Config.REWARD_INFO["target_warehouses"]:
                    self.iot_sensors[f_name]["warehouses"][w_name] = self.reward_meters[w_name].diff(-1,-2)
                else:
                    self.iot_sensors[f_name]["warehouses"][w_name] = self.factories[f_name].warehouses[w_name].storage.level
            for m_name in Config.MACHINES_UNDER_MONITOR[f_name]:
                self.iot_sensors[f_name]["machines"][m_name] = self.factories[f_name].machines[m_name].current_production_rate
            for l_name, _ in Config.LORRIES.items():
                self.iot_sensors["lorries"][l_name] = self.lorries[l_name].get_operation_status()
        return self.iot_sensors
