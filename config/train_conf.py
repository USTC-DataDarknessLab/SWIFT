import json




class GlobalConfig:

    conf_path = '/root/swift/config/train_conf'

    

    def __init__(self):
        self.data_incre = True
        self.use_bucket = True
        self.memory_disk = False
        self.model_eval = False
        self.node_cache = False
        self.node_reorder = False
        self.node_simple_cache = False
        self.edge_reorder = True
        self.epoch = -1
        self.pre_sample_size = -1


        filename = f'{self.conf_path}/{self.conf}'
        with open(filename, 'r') as file:
            self.config_data = json.load(file)
        
        for key, value in self.config_data.items():
            setattr(self, key, value)


