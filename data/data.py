from scripts.time_log import time_log_module as tlm

class data():
    def __init__(self, logger, data_path, dataset_loading_size):
        self.logger = logger
        self.data_path = data_path
        self.dataset_loading_size = dataset_loading_size
        self.data = ""
    
    def load_data(self):
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.data = f.read(self.dataset_loading_size)
            self.logger.log(f"Data loaded successfully from {self.data_path}.", v=True, Wh=True, mention=False)
        except Exception as e:
            self.logger.log(f"Error loading data from {self.data_path}: {e}", v=False, Wh=True, mention=True)
            raise ValueError(f"{tlm()} Error loading data from {self.data_path}: {e}")
        
        return self.data
