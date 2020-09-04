# Experiment manager
Single file experiment manager for `tf.keras`.

The experiment manager, is here to automatically keep a tidy model checkpoints, performance files for your hyperparameter search. It's doing so by maintaining a consistent folder hierarchy based on the hash of your hyperparameters.

This tool automatically: 
- generate a tree structure for each experiment under a specific identifier and the current date
- generate callback for `tf.keras` to ensure that training logs and model checkpoints are written in the same directory

Let's go through an example to see **how it works**. First let's define our hyperparemeters in a separated JSON file:

```python
params = {
    'debug': False,
    'training': {
      'batch_size': 128,
      'epochs': 12,
      'learning-rate': 0.008  
    },
    'processing': {
      'width': 128,
      'height':  128,
    },
    'model': {
      'conv_0': 32,
      'conv_1': 64,
      'kernel_0': (3,3),
      'kernel_1': (3,3),
      'pool_size': (2,2),
      'dropout_0': 0.25,
      'dense': 128,
      'dropout_1': 0.5,
      'num_classes': 10
    },
    'comment': 'VGG net',
    'author': 'john doe'
}
```

Then build your model using the dict above:
```python
model = get_model(params) # you need to create this function
# ...
expm = ExperimentManager(
      exp_base_dir='notedetection', # You're experiment name, get_model version, whatever works for you
      monitored_param_keys=['training', 'processing', 'model'])      
callbacks_p = expm.prepare(parameters)
model.fit(..., 
          callbacks=callbacks_p, 
          ...)

```

Let's say it's the 3rd of March 2020 at 15:52, running at this time the script above will generate the following folder structure:

```
notedetection
└── exp-52663881-19659650-2212381
    └── run--20-03-03--15-52
        ├── hyperparameters.json
        ├── performances.json
        └── models
            ├── model.{epoch}-{loss}.h5
            └── ...
```

Where `hash(params['training']) == 52663881`, `hash(params['processing']) == 19659650` and `hash(params['model']) == 2212381`. 
This unique identifier ensure that running the same experiment twice will create a `run--…` directory and changing it is also easier to check where the difference lies between two experiments:

```
notedetection
└── exp-52663881-19659650-2212381 
└── exp-79023743-19659650-2212381 ==> Same model and processing steps but different training parameter
```

Under the hood, this simple tool creates `tf.keras.callbacks` objects in an unifed manner.
