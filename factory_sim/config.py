class Config:
    PRODUCTION_RATES = [2, 4, 6, 8, 10]
    POWER_LIST = [6., 12., 18., 24., 30.]
    UNIT_ENERGY_COST = 1.0
    STORAGE_CAPACITY = 100
    INIT_STORAGE = 0

    PRODUCTION_UNIT_TIME = 5
    STATE_TRANSITION_TIME = 5
    DECISION_INTERVAL = 15

    N_STEPS = 500

    FACTORY_1 = {
        "warehouses": {
            "in_warehouse_1": {},
            "export_warehouse_1": {"init_storage": 0, "storage_capacity": STORAGE_CAPACITY},
        },

        "machines": {
            "machine_1": {"production_rates": PRODUCTION_RATES, "power_list": POWER_LIST},
        },

        "products": {
            "product_1": {"material_costs_per_unit": [1], "material_price_per_unit": [5.], "production_line": (["in_warehouse_1"], "machine_1", ["export_warehouse_1"])},
        },

    }

    FACTORY_2 = {
        "warehouses": {
            "import_warehouse_1": {"init_storage": INIT_STORAGE, "storage_capacity": STORAGE_CAPACITY},
            "in_warehouse_2": {},
            "out_warehouse_2": {"init_storage": INIT_STORAGE, "storage_capacity": STORAGE_CAPACITY},
            "export_warehouse_2": {"init_storage": 0, "storage_capacity": STORAGE_CAPACITY},
            "export_warehouse_1-2": {"init_storage": 0, "storage_capacity": STORAGE_CAPACITY},
        },

        "machines": {
            "machine_2": {"production_rates": PRODUCTION_RATES, "power_list": POWER_LIST},
            "machine_1-2": {"production_rates": PRODUCTION_RATES, "power_list": POWER_LIST},
        },

        "products": {
            "product_2": {"material_costs_per_unit": [1], "material_price_per_unit": [10.], "production_line": (["in_warehouse_2"], "machine_2", ["out_warehouse_2", "export_warehouse_2"])},
            "product_1-2": {"material_costs_per_unit": [2, 1], "material_price_per_unit": [0., 0.], "production_line": (["import_warehouse_1", "out_warehouse_2"], "machine_1-2", ["export_warehouse_1-2"])},
        },
        
    }

    FACTORY_3 = {
        "warehouses": {
            "in_warehouse_3":{},
            "out_warehouse_3a": {"init_storage": INIT_STORAGE, "storage_capacity": STORAGE_CAPACITY},
            "out_warehouse_3b": {"init_storage": INIT_STORAGE, "storage_capacity": STORAGE_CAPACITY},
            "import_warehouse_1-2": {"init_storage": INIT_STORAGE, "storage_capacity": STORAGE_CAPACITY},
            "import_warehouse_2": {"init_storage": INIT_STORAGE, "storage_capacity": STORAGE_CAPACITY},
            "export_warehouse_2-3": {"init_storage": 0, "storage_capacity": STORAGE_CAPACITY},
            "out_warehouse_1-2-3": {"init_storage": 0},
        },

        "machines": {
            "machine_3": {"production_rates": PRODUCTION_RATES, "power_list": POWER_LIST},
            "machine_1-2-3": {"production_rates": PRODUCTION_RATES, "power_list": POWER_LIST},
            "machine_2-3": {"production_rates": PRODUCTION_RATES, "power_list": POWER_LIST},
        },

        "products": {
            "product_3": {"material_costs_per_unit": [1], "material_price_per_unit": [15.], "production_line": (["in_warehouse_3"], "machine_3", ["out_warehouse_3a", "out_warehouse_3b"])},
            "product_1-2-3": {"material_costs_per_unit": [2, 1], "material_price_per_unit": [0., 0.], "production_line": (["import_warehouse_1-2", "out_warehouse_3a"], "machine_1-2-3", ["out_warehouse_1-2-3"])},
            "product_2-3": {"material_costs_per_unit": [1, 2], "material_price_per_unit": [0., 0.], "production_line": (["import_warehouse_2", "out_warehouse_3b"], "machine_2-3", ["export_warehouse_2-3"])},
        },
        
    }

    FACTORY_4 = {
        "warehouses": {
            "in_warehouse_4": {},
            "out_warehouse_4": {"init_storage": INIT_STORAGE, "storage_capacity": STORAGE_CAPACITY},
            "import_warehouse_2-3": {"init_storage": INIT_STORAGE, "storage_capacity": STORAGE_CAPACITY},
            "out_warehouse_2-3-4": {"init_storage": 0},
        },

        "machines": {
            "machine_4": {"production_rates": PRODUCTION_RATES, "power_list": POWER_LIST},
            "machine_2-3-4": {"production_rates": PRODUCTION_RATES, "power_list": POWER_LIST},
        },

        "products": {
            "product_4": {"material_costs_per_unit": [1], "material_price_per_unit": [20.], "production_line": (["in_warehouse_4"], "machine_4", ["out_warehouse_4"])},
            "product_2-3-4": {"material_costs_per_unit": [1, 4], "material_price_per_unit": [0., 0.], "production_line": (["import_warehouse_2-3", "out_warehouse_4"], "machine_2-3-4", ["out_warehouse_2-3-4"])},
        },

    }

    LORRIES = {
            "lorry_1": {"capacity": 80, "min_load": 40, "average_delay": 25, "route": (["factory_1", "export_warehouse_1"], ["factory_2", "import_warehouse_1"])},
            "lorry_2": {"capacity": 80, "min_load": 40, "average_delay": 25, "route": (["factory_2", "export_warehouse_2"], ["factory_3", "import_warehouse_2"])},
            "lorry_1-2": {"capacity": 40, "min_load": 20, "average_delay": 25, "route": (["factory_2", "export_warehouse_1-2"], ["factory_3", "import_warehouse_1-2"])},
            "lorry_2-3": {"capacity": 40, "min_load": 20, "average_delay": 25, "route": (["factory_3", "export_warehouse_2-3"], ["factory_4", "import_warehouse_2-3"])},
        }

    WAREHOUSES_UNDER_MONITOR = {
        "factory_1": ["export_warehouse_1"],
        "factory_2": ["import_warehouse_1", "out_warehouse_2", "export_warehouse_2", "export_warehouse_1-2"],
        "factory_3": ["out_warehouse_3a", "out_warehouse_3b", "import_warehouse_1-2", "import_warehouse_2", "export_warehouse_2-3", "out_warehouse_1-2-3"],
        "factory_4": ["out_warehouse_4", "import_warehouse_2-3", "out_warehouse_2-3-4"]
    }

    MACHINES_UNDER_MONITOR = {
        "factory_1": ["machine_1"],
        "factory_2": ["machine_2", "machine_1-2"],
        "factory_3": ["machine_3", "machine_1-2-3", "machine_2-3"],
        "factory_4": ["machine_4", "machine_2-3-4"]
    }

    WAREHOUSES_LIST = ["export_warehouse_1", 
                        "import_warehouse_1", "out_warehouse_2", "export_warehouse_2", "export_warehouse_1-2", 
                        "out_warehouse_3a", "out_warehouse_3b", "import_warehouse_1-2", "import_warehouse_2", "export_warehouse_2-3", "out_warehouse_1-2-3", 
                        "out_warehouse_4", "import_warehouse_2-3", "out_warehouse_2-3-4"]

    VOLUME_COEFF = [1, 
                    1, 1, 1, 3, 
                    1, 1, 3, 1, 3, 0, 
                    1, 3, 0]
    
    ENERGY_COST_COEFF = {"machine_1": 0.25, 

                        "machine_2": 0.25, 
                        "machine_1-2": 1.0, 
                        
                        "machine_3": 0.25, 
                        "machine_1-2-3": 1.0, 
                        "machine_2-3": 1.0, 
                        
                        "machine_4": 0.25, 
                        "machine_2-3-4": 1.0}

    OBSERVATION_WAREHOUSE = {
        "machine_1": ["export_warehouse_1", "import_warehouse_1"],

        "machine_2": ["import_warehouse_1", "out_warehouse_2", "export_warehouse_2", "export_warehouse_1-2", "import_warehouse_2"],
        "machine_1-2": ["import_warehouse_1", "out_warehouse_2", "export_warehouse_2", "export_warehouse_1-2", "import_warehouse_1-2"],

        "machine_3": ["out_warehouse_3a", "out_warehouse_3b", "import_warehouse_1-2", "import_warehouse_2", "export_warehouse_2-3"],
        "machine_1-2-3": ["out_warehouse_3a", "out_warehouse_3b", "import_warehouse_1-2"],
        "machine_2-3": ["out_warehouse_3a", "out_warehouse_3b", "import_warehouse_2", "export_warehouse_2-3", "import_warehouse_2-3"],

        "machine_4": ["out_warehouse_4", "import_warehouse_2-3"],
        "machine_2-3-4": ["out_warehouse_4", "import_warehouse_2-3"],
    }

    OBSERVATION_LORRY = {
        "machine_1": ["lorry_1"],

        "machine_2": ["lorry_1", "lorry_2"],
        "machine_1-2": ["lorry_1", "lorry_1-2"],

        "machine_3": ["lorry_2", "lorry_1-2"],
        "machine_1-2-3": ["lorry_1-2"],
        "machine_2-3": ["lorry_2", "lorry_2-3"],

        "machine_4": ["lorry_2-3"],
        "machine_2-3-4": ["lorry_2-3"],

    }

    REWARD_INFO = {
        "target_warehouses": ["out_warehouse_1-2-3", "out_warehouse_2-3-4"],
        "prices": [100., 200.],
    }