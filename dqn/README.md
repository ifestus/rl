# DQN Implementation


## How to run

1. `pip install` at the root level of the repository.
2. `source venv/bin/activate` to activate your virtual environment
3. `python run_environment.py` will run the model against Atari2600


## Model

* `./dqn_model.py` contains the implementation of a DQN class (tf model).
* `./dqn.py` is the interface between a program and the `dqn_model`.


## TF_GRAPH

`TF_GRAPH` is set up for the DQN model runs. These are written into the `./tf_graph` directory.


## TODO

Currently working on a simpler implementation for testing the model against `Classic` environments.
