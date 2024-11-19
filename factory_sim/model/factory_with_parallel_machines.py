import simpy
import random
import numpy as np
from factory_sim.config import Config

class ParallelMachineState:
    def __init__(self, n_parallel_machines):
        self.n_states = 5
        self.n_parallel_machines = n_parallel_machines
        self.recovery_time = 2
        self.l = 0.33333
        self.state_transition_matrix = [
            [1. - self.l, self.l, 0, 0, 0],
            [0, 1. - self.l, self.l, 0, 0],
            [0, 0, 1. - self.l, self.l, 0],
            [0, 0, 0, 1. - self.l, self.l]]
        self.current_machines_state = [0 for _ in range(self.n_parallel_machines)]
        self.in_maintenance_time_counter = [0 for _ in range(self.n_parallel_machines)]
    
    def transit(self):
        for machine_id in self.n_parallel_machines:
            if self.current_machines_state[machine_id] == 4:
                self.in_maintenance_time_counter[machine_id] += 1
                if self.in_maintenance_time_counter[machine_id] == self.recovery_time:
                    self.current_machines_state[machine_id] = 0
            else:
                transition_possibilities = self.state_transition_matrix[self.current_machines_state[machine_id]]
                self.current_machines_state[machine_id] = random.choices(range(self.n_states), weights=transition_possibilities)[0]
    
    def reset(self):
        self.current_machines_state = [0 for _ in range(self.n_parallel_machines)]
        self.in_maintenance_time_counter = [0 for _ in range(self.n_parallel_machines)]
    
    def in_operation(self, machine_id):
        return self.in_maintenance_time_counter[machine_id] == 0
    
    def get_operation_status(self):
        return self.current_machines_state.count(4)


class Warehouse(simpy.Resource):
    def __init__(self, env, name, storage_capacity=float("inf"), init_storage=float("inf")):
        super().__init__(env, 1)
        self.env = env
        self.name = name
        self.storage_capacity = storage_capacity
        self.init_storage = init_storage
        self.input_num_memory = data_memory(6)
        self.output_num_memory = data_memory(6)
        self.input_count = 0
        self.output_count = 0
        self.storage = simpy.Container(env, storage_capacity, init_storage)

    def is_full(self, products_produced):
        return self.storage.capacity < (self.storage.level + products_produced)

    def is_empty(self, material_cost):
        return (self.storage.level - material_cost) < 0

    def get_storage_status(self):
        self.input_num_memory.update(self.input_count)
        self.output_num_memory.update(self.output_count)
        self.input_count, self.output_count = 0, 0
        return [self.storage.level]

    def reset(self):
        self.input_num_memory.reset()
        self.output_num_memory.reset()
        self.input_count = 0
        self.output_count = 0
        self.storage = simpy.Container(self.env, self.storage_capacity, self.init_storage)


class data_memory:
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = [0 for _ in range(self.max_length)]
    
    def update(self, value):
        self.data.append(value)
        self.data = self.data[-self.max_length:]

    def reset(self):
        self.data = [0 for _ in range(self.max_length)]


class Product:
    def __init__(self, name, material_costs_per_unit, material_price_per_unit, production_line):
        self.name = name
        self.material_costs_per_unit = material_costs_per_unit
        self.in_warehouse_names = production_line[0]
        self.machine_name = production_line[1]
        self.out_warehouse_names = production_line[2]
        self.material_price_per_unit = material_price_per_unit


class Factory:
    def __init__(self, env, name, machines, warehouses, products):
        self.env = env
        self.name = name
        self.machines = machines
        self.warehouses = warehouses
        self.products = products

    def run(self):
        for product_name in self.products:
            product = self.products[product_name]
            in_warehouses = [self.warehouses[name] for name in product.in_warehouse_names]
            machine = self.machines[product.machine_name]
            out_warehouses = [self.warehouses[name] for name in product.out_warehouse_names]
            self.env.process(machine.produce(product, in_warehouses, out_warehouses))
    
    def take_actions(self, actions):
        for i, name in enumerate(self.machines):
            self.machines[name].set_production_rate(actions[i])


class Machine:
    PRODUCTION_UNIT_TIME = Config.PRODUCTION_UNIT_TIME
    DECISION_INTERVAL = Config.DECISION_INTERVAL
    STATE_TRANSITION_TIME = Config.STATE_TRANSITION_TIME

    def __init__(self, env, name, production_rates, power_list):
        self.env = env
        self.name = name
        self.production_rates = production_rates
        self.current_production_rate = production_rates[0]
        self.power_list = power_list
        self.current_power = power_list[0]
        self.time_counter = 0
        self.num_item_in_production = 0
        self.material_total_price = 0
        self.num_item_produced = 0
        assert len(production_rates) == len(power_list)

    def check_input(self, product, in_warehouses):
        ready_for_production_num = min(wh.storage.level//product.material_costs_per_unit[i] if wh.storage.capacity != float('inf') else float('inf') for i, wh in enumerate(in_warehouses))
        return ready_for_production_num

    def output_ready(self, product, out_warehouses):
        full_output = all(wh.is_full(self.num_item_in_production) for i, wh in enumerate(out_warehouses))
        return not full_output
    
    def set_production_rate(self, action):
        self.current_production_rate = self.production_rates[action]
        self.current_power = self.power_list[action]
        
    def get_energy_cost(self):
        num_item_produced = self.num_item_produced
        self.num_item_produced = 0
        return  (1 - num_item_produced / (Machine.DECISION_INTERVAL / Machine.PRODUCTION_UNIT_TIME * self.current_production_rate)) * self.current_power

    def get_material_total_price(self):
        material_total_price = self.material_total_price
        self.material_total_price = 0
        return material_total_price

    def produce(self, product, in_warehouses, out_warehouses):
        while True:
            if self.time_counter % Machine.PRODUCTION_UNIT_TIME == Machine.PRODUCTION_UNIT_TIME - 1:
                if self.num_item_in_production != 0:
                    if self.output_ready(product, out_warehouses):
                        storage_level_buffer = []
                        for wh in out_warehouses:
                            storage_level_buffer.append(wh.storage.level)
                        out_warehouse = out_warehouses[np.argmin(storage_level_buffer)]

                        with out_warehouse.request() as request:
                            yield request
                            yield out_warehouse.storage.put(self.num_item_in_production)
                        out_warehouse.input_count += self.num_item_in_production
                        self.num_item_in_production = 0
            
            if self.time_counter % Machine.PRODUCTION_UNIT_TIME == 0: 
                if self.num_item_in_production == 0:
                    ready_for_production_num = self.check_input(product, in_warehouses)
                    if ready_for_production_num != 0:
                        self.num_item_in_production = min(ready_for_production_num, self.current_production_rate)
                        self.num_item_produced += self.num_item_in_production
                        for i, in_warehouse in enumerate(in_warehouses):
                            material_cost = product.material_costs_per_unit[i] * self.num_item_in_production
                            self.material_total_price += self.num_item_in_production * product.material_price_per_unit[i]
                            with in_warehouse.request() as request:
                                yield request
                                yield in_warehouse.storage.get(material_cost)
                            in_warehouse.output_count += material_cost

            self.time_counter += 1
            yield self.env.timeout(1)

    def reset(self):
        self.num_item_in_production = 0
        self.time_counter = 0
        self.material_total_price = 0
        self.num_item_produced = 0