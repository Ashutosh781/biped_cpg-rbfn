import os
import sys
import numpy as np

# Add project root to the python path
sys.path.append(os.path.dirname(__file__))

from evolutionary.individual import Models
from evolutionary.neuroevolution import NeuroEvolution
from rl.rlpibb import RlPibb


def neuro_evolution_train(model_type:str, env_type: str, fixed_centres: bool, generations: int, max_steps: int, gen_size: int,
                          elite_size: int, load_elite: bool=False, mean: float=1.0, std: float=0.001):
    """Train a model using neuroevolution"""

    try:
        # Check model type
        if model_type not in ["FC", "CPG-FC", "RBFN-FC", "CPG-RBFN"]:
            raise ValueError("Model type not supported")

        # Initialize neuroevolution
        neuro_evolution = NeuroEvolution(model_type=model_type, env_type=env_type, fixed_centres=fixed_centres, generations=generations, max_steps=max_steps,
                                         gen_size=gen_size, mean=mean, std=std, elite_size=elite_size, load_elite=load_elite)

        # Run neuroevolution
        print("Running Neuro evolution training...")
        neuro_evolution.run(verbose=True)

        # Get path to save data
        model_path = os.path.join(os.getcwd(), "data", model_type)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save results
        print("TRAINING COMPLETED...")
        neuro_evolution.save(model_path)

        # Close environment
        neuro_evolution.env.close()

        # Get plots
        neuro_evolution.get_plots(model_path, is_show=False)

    except KeyboardInterrupt:
        print("TRAINING INTERRUPTED !!")

        # Get path to save data
        model_path = os.path.join(os.getcwd(), "data", model_type)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save data
        neuro_evolution.save(model_path)

        # Close environment
        neuro_evolution.env.close()

        # Get plots
        neuro_evolution.get_plots(model_path, is_show=False)

        # Exit
        sys.exit()

def rl_pibb_train(env_type: str, epochs: int, max_steps: int, rollout_size: int, norm_constant: float, variance: float, decay:float):
    """Train a model using RL-PIBB"""

    try:
        # Initialize RL-PIBB
        rl_pibb = RlPibb(env_type, epochs, max_steps, rollout_size, norm_constant, variance, decay)

        # Run RL-PIBB
        print("Running RL-PIBB training...")
        rl_pibb.run(verbose=True)

        # Get path to save data
        model_path = os.path.join(os.getcwd(), "data", "RL-PIBB")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save results
        print("TRAINING COMPLETED...")
        rl_pibb.save(model_path)

        # Close environment
        rl_pibb.env.close()

        # Get plots
        rl_pibb.get_plots(model_path, is_show=False)

    except KeyboardInterrupt:
        print("TRAINING INTERRUPTED !!")

        # Get path to save data
        model_path = os.path.join(os.getcwd(), "data", "RL-PIBB")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save data
        rl_pibb.save(model_path)

        # Close environment
        rl_pibb.env.close()

        # Get plots
        rl_pibb.get_plots(model_path, is_show=False)

        # Exit
        sys.exit()


if __name__ == "__main__":

    #Gym environment
    env_type = "HalfCheetah-v4"

    ## MODEL TYPE
    models = Models()
    # model_type = models.FC_MODEL
    # model_type = models.CPG_FC_MODEL
    # model_type = models.RBFN_FC_MODEL
    model_type = models.CPG_RBFN_MODEL

    # NEUROEVOLUTION PARAMS
    fixed_centres = False
    generations = 100
    max_steps = 1000
    gen_size = 10
    elite_size = 10
    load_elite = False
    mean = 1.0
    std = 0.001

    # Run neuroevolution
    neuro_evolution_train(model_type=model_type, env_type=env_type, fixed_centres=fixed_centres, generations=generations, max_steps=max_steps,
                          gen_size=gen_size, mean=mean, std=std, elite_size=elite_size, load_elite=load_elite)

    # RL-PIBB PARAMS
    epochs = 1000
    max_steps = 1000
    rollout_size = 10
    norm_constant = 10.0
    variance = 1.0
    decay = 0.99

    # Run RL-PIBB
    rl_pibb_train(env_type, epochs, max_steps, rollout_size, norm_constant, variance, decay)