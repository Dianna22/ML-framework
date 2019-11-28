# -*- coding: utf-8 -*-
from utils import file_helpers as fh
import torch

class Experiment(object):
    """Holds all the experiment parameters and provides helper functions."""
    def __init__(self, e_id):
        self.id = e_id
        
    def restore_model(self):
        if self.restore_checkpoint:
            checkpoint = self.model.args.dirs.checkpoint
            if checkpoint is not None:
                last_checkpoint = fh.get_last_checkpoint(checkpoint)
                print(f"Loading latest checkpoint: {last_checkpoint}")
                self.model = torch.load(last_checkpoint)    

    def setup(self):
        self.restore_model()
        return self
                

    ### DECLARATIVE API ###

    def with_data(self, data):
        self.data = data
        return self

    def with_config(self, config):
        self.config = config.copy()
        return self

    def override(self, config):
        self.config.update(config)
        return self

    def with_model(self, model):
        self.model = model
        return self
    #### END API ######
    
    @property
    def experiment_name(self):
        return f'E-{self.id}_M-{self.model.id}'

    """ Dirs
    - *_dir - full path to dir
    """
    @property
    def experiments_dir(self):
        return "experiments"


    @property
    def latest_checkpoint(self):
        """Look inside the saved models dir and retrieve the latest checkpoint """
        import re
        import os
        list_of_files = os.listdir(self.models_dir)
        # list_of_files = ["checkpoint_1","checkpoint_10","checkpoint_2", "checkpoint_22"]

        def extract_number(f):
            s = re.findall("\d+$",f)
            return int(s[0])

        n = max([extract_number(f) for f in list_of_files]) if list_of_files else None
        if n is None:
            return None

        p = os.path.join(self.model.args.dirs.checkpoint, f'checkpoint_{n}')
        return os.path.join(p, f'checkpoint-{n}')

    def run(self):
        pass
        
        