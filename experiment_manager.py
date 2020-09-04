import datetime
import json
from pathlib import Path
import os
import glob
import errno
import hashlib

import pandas as pd
from timeit import default_timer as timer

from tensorflow.keras.callbacks import *
from collections import OrderedDict


class WriteBestPerformances(Callback):
    """
    This callback read at the end of training the progress
    csv file (located at csv_path) and write the 
    lowest loss epoch line into a 
    separated json file (located at output_path).
    """
    def __init__(self, 
               csv_path, 
               output_path,
               watched_metric='loss'):
        self.csv_path = csv_path
        self.output_path = output_path
        self.watched_metric = watched_metric

    def on_train_begin(self, logs={}):
        self.start = timer()

    def on_train_end(self, logs={},):
        elapsed = timer() - self.start
        df = pd.read_csv(self.csv_path)
        df['elapsed'] = elapsed
        min_loss_index = (df[self.watched_metric] == df[self.watched_metric].min())
        df[min_loss_index].to_json(self.output_path, orient='records')


class ExperimentManager():
    """
        This class aims at sorting experiments by creating
        folder on the fly and create callbacks.

        Minimal usage: callbacks = ExperimentManager().prepare(params)
    """
    def __init__(self, 
                 exp_base_dir=None,
                 monitored_param_keys=['model', 'training'],
                 checkpoint_params=None):
        self.params = None

        if(exp_base_dir is None):
            self.base_experiment_path = 'experiments'
        else:
            self.base_experiment_path = exp_base_dir

        self.run_path = None
        self.run_id = None
        self.experiment_id_folder = None
        self.monitored_param_keys = monitored_param_keys
        
        if(checkpoint_params is None):
            self.checkpoint_params = {
                'mode': 'auto',
                'monitor': 'val_loss'
            }
        else:
            self.checkpoint_params = checkpoint_params

    def resume_run(self, experiment_id_folder, run_id_folder):
        """
            Allow to resume an experiment identified by its 
            experiment_id and its run_id
        """
        supposed_run = Path(self.base_experiment_path) / experiment_id_folder / run_id_folder
        if(supposed_run.exists()):
            self.run_id = run_id_folder
            self.experiment_id_folder = experiment_id_folder
            latest_checkpoint = max(glob.glob(str(supposed_run / 'models' /'*')), key=os.path.getmtime)
            print('Latest checkpoint is: {}'.format(latest_checkpoint))
            return latest_checkpoint
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(supposed_run))

    def get_experiment_id(self):
        """
          Hash model and training parameters separately into two 8 digits strings
          and return the experiment identifier
        """
        if(self.experiment_id_folder is None):
            #stackoverflow.com/a/16008760
            hash_8 = lambda x: str(int(hashlib.sha256(str(json.dumps(x, sort_keys=True)).encode('utf-8')).hexdigest(), base=16))[:8]
            
            id_hash = ''
            for k in self.monitored_param_keys:
                id_hash += '-{}'.format(hash_8(self.params[k]))
                
            return 'exp-{}'.format(id_hash)
        else:
            return self.experiment_id_folder

    def get_run_id(self):
        """
          Returns the identifier of the current run
          by formatting the dateime.now() object
        """
        if(self.run_id is not None):
            return self.run_id
        else:
            return 'run--{}'.format(datetime.datetime.now().strftime('%y-%m-%d--%H-%M'))

    # Write params to file
    def write_parameters(self):
        """
          Serialize the hyperparameters file into a json file
        """
        if(self.experiment_id_folder is None):
            # Write only if we don't resume training
            with open(str(self.get_param_path()), 'w', encoding='utf-8') as f:
                json.dump(self.params, f, sort_keys=True, indent=2)

    def make_experiment_path(self):
        """
          Create the experiment directory
        """
        if(self.params['debug']):
            run_path = self.get_debug_path()
        else:
            run_path = self.get_run_path()

        run_path.mkdir(parents=True, exist_ok=True)
        self.run_path = run_path.resolve()

        print('Now everything is happening in {}'.format(self.run_path))
        self.get_best_model_path().mkdir(parents=True, exist_ok=True)
        self.write_parameters()

    def get_log_error_path(self):
        return self.run_path / 'errors.log'

    def get_csv_path(self):
        return self.run_path / 'training-logs.csv'

    def get_best_model_path(self):
        return self.run_path / 'models' 

    def get_param_path(self):
        return self.run_path / 'hyperparameters.json'

    def get_debug_path(self):
        return Path(self.base_experiment_path) / 'debug'

    def get_run_path(self):
        exp_id = self.get_experiment_id()
        run_id = self.get_run_id()
        return Path(self.base_experiment_path) / exp_id / run_id

    def get_best_perf_path(self):
        return self.run_path / 'performances.json'

    def set_parameter_files(self, params):
        self.params = OrderedDict(params) 

    def prepare(self, params):
        self.set_parameter_files(params)
        self.make_experiment_path()

        csv_callback = CSVLogger(str(self.get_csv_path()), 
                                 separator=',', 
                                 append=True)

        template_path = 'model.{epoch:02d}-{'+self.checkpoint_params['monitor']+':.4f}.hdf5'
        model_callback = ModelCheckpoint(filepath=str(self.get_best_model_path() / template_path), 
                                         save_best_only=True,
                                         **self.checkpoint_params)
        
        best_performances = WriteBestPerformances(csv_path=str(self.get_csv_path()), 
                                                  output_path=str(self.get_best_perf_path()))

        callbacks = [csv_callback, model_callback, best_performances]

        return callbacks
