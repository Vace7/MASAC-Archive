import numpy as np

class Logistic_center(): 
    def __init__(self, env, name, factories, lorries):
        self.env = env
        self.name = name
        self.factories = factories
        self.lorries = lorries
        self.run()
    def run(self):
        for name in self.lorries:
            lorry = self.lorries[name]
            in_warehouse = self.factories[lorry.src[0]].warehouses[lorry.src[1]]
            out_warehouse = self.factories[lorry.dst[0]].warehouses[lorry.dst[1]]
            self.env.process(lorry.transport(in_warehouse, out_warehouse))
          
class Lorry():
    def __init__(self, env, name, capacity, min_load, average_delay, route):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.min_load = min_load
        self.average_delay = average_delay
        self.src = route[0]
        self.dst = route[1]
        self.cargo_num = 0
        self.time_counter = 0
        self.delay = 100
    
    def in_operation(self):
        return self.cargo_num != 0

    def get_operation_status(self):
        normalized_num = self.cargo_num / self.capacity
        tp_progress = min(1, self.time_counter / self.delay)
        return [normalized_num, tp_progress]

    def transport(self, in_warehouse, out_warehouse):
        while True:
            if self.time_counter == 0:
                self.delay = self.average_delay + self.sample_from_poisson(4)
                if not in_warehouse.is_empty(self.min_load):
                    self.cargo_num = min(in_warehouse.storage.level, self.capacity)
                    with in_warehouse.request() as request:
                        yield request
                        yield in_warehouse.storage.get(self.cargo_num)
                        in_warehouse.output_count += self.cargo_num

            if self.time_counter >= self.delay:
                if not out_warehouse.is_full(self.cargo_num):
                    with out_warehouse.request() as request:
                        yield request
                        yield out_warehouse.storage.put(self.cargo_num)
                        out_warehouse.input_count += self.cargo_num
                    self.cargo_num = 0
                    self.time_counter = 0

            if self.in_operation():
                self.time_counter += 1
            yield self.env.timeout(1)
        
    def reset(self):
        self.cargo_num = 0
        self.time_counter = 0
        self.delay = 100

    def sample_from_poisson(self, lmd):
        l = np.exp(-lmd)
        k = 0
        p = 1
        while p > l:
            p = p * np.random.rand()
            k += 1 
        return k - 1


        

