import os
import sys
import numpy as np

# Add project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from evolutionary.individual import Models
from evolutionary.neuroevolution import NeuroEvolution
from rl.rlpibb import RlPibb


def neuro_evolution(model_type:str, env_type: str, generations: int, max_steps: int, gen_size: int, mean: float=1.0, std: float=0.001):
    """Train a model using neuroevolution"""

    try:
        # Check model type
        if model_type not in ["FC", "CPG-FC", "RBFN-FC", "CPG-RBFN"]:
            raise ValueError("Model type not supported")

        # Initialize neuroevolution
        neuro_evolution = NeuroEvolution(model_type, env_type, generations, max_steps, gen_size, mean, std)

        # Run neuroevolution
        print("Running Neuro evolution training...")
        neuro_evolution.run(verbose=True)

    except KeyboardInterrupt:
        print("TRAINING INTERRUPTED !!")

        # Get path to save data
        model_path = os.path.join(os.getcwd(), "data", model_type)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save data
        neuro_evolution.save_data(model_path)

        # Close environment
        neuro_evolution.env.close()

        # Get plots
        neuro_evolution.get_plots(is_show=True)

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
    # model_type = models.RL_PIBB

    # NEUROEVOLUTION PARAMS
    generations = 500
    max_steps = 1000
    gen_size = 10
    mean = 1.0
    std = 0.001

    # Run neuroevolution
    neuro_evolution(model_type, env_type, generations, max_steps, gen_size, mean, std)