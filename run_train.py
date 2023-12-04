import os
import sys
import numpy as np

# Add project root to the python path
sys.path.append(os.path.dirname(__file__))

from evolutionary.individual import Models
from evolutionary.neuroevolution import NeuroEvolution
from rl.rlpibb import RlPibb


def neuro_evolution_train(model_type:str, env_type: str, fixed_centres: bool, generations: int, max_steps: int, gen_size: int,
                          elite_size: int, load_elite: bool=False, alt_cpgs: bool=False, add_noise: bool=False, mean: float=1.0, std: float=0.001):
    """Train a model using neuroevolution"""

    try:
        # Check model type
        if model_type not in ["FC", "CPG-FC", "RBFN-FC", "CPG-RBFN"]:
            raise ValueError("Model type not supported")

        # Initialize neuroevolution
        neuro_evolution = NeuroEvolution(model_type=model_type, env_type=env_type, fixed_centres=fixed_centres, generations=generations, max_steps=max_steps,
                                         gen_size=gen_size, mean=mean, std=std, elite_size=elite_size, load_elite=load_elite, alt_cpgs=alt_cpgs, add_noise=add_noise)

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
        model_path = os.path.join(os.getcwd(), "data", env_type, model_type, "not fixed")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        #Set new path to save files if fixed centers are selected
        if fixed_centres:
            model_path = os.path.join(os.getcwd(), "data", env_type, model_type, "fixed")
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        
        elif alt_cpgs and add_noise:
            model_path = os.path.join(os.getcwd(), "data", env_type, model_type, "noisy_alt_cpgs")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

        elif alt_cpgs:
            model_path = os.path.join(os.getcwd(), "data", env_type, model_type, "alt_cpgs")
            if not os.path.exists(model_path):
                os.makedirs(model_path)

        elif add_noise:
            model_path = os.path.join(os.getcwd(), "data", env_type, model_type, "add_noise")
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

def rl_pibb_train(env_type: str, epochs: int, max_steps: int, rollout_size: int, norm_constant: float,
                  variance: float, decay:float, alt_cpgs: bool, add_noise: bool, test_case: int):
    """Train a model using RL-PIBB"""

    try:
        # Initialize RL-PIBB
        rl_pibb = RlPibb(env_type, epochs, max_steps, rollout_size, norm_constant, variance, decay, alt_cpgs, add_noise, test_case)

        # Run RL-PIBB
        print("Running RL-PIBB training...")
        rl_pibb.run(verbose=True)

        # Get path to save data
        model_path = os.path.join(os.getcwd(), "data", "RL-PIBB", f"set_{test_case}_alt{int(alt_cpgs)}_noise{int(add_noise)}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save results
        print("TRAINING COMPLETED...")
        rl_pibb.save(model_path)

        # Close environment
        rl_pibb.env.close()

        # Get plots
        # rl_pibb.get_plots(model_path, is_show=False)

    except KeyboardInterrupt:
        print("TRAINING INTERRUPTED !!")

        # Get path to save data
        model_path = os.path.join(os.getcwd(), "data", "RL-PIBB", f"set_{test_case}_alt{int(alt_cpgs)}_noise{int(add_noise)}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Save data
        rl_pibb.save(model_path)

        # Close environment
        rl_pibb.env.close()

        # Get plots
        # rl_pibb.get_plots(model_path, is_show=False)

        # Exit
        sys.exit()


if __name__ == "__main__":

    #Gym environment
    env_type = "HalfCheetah-v4"
    # env_type = "Walker2d-v4"

    ## MODEL TYPE
    models = Models()
    # model_type = models.FC_MODEL
    # model_type = models.CPG_FC_MODEL
    # model_type = models.RBFN_FC_MODEL
    model_type = models.CPG_RBFN_MODEL

    # NEUROEVOLUTION PARAMS
    fixed_centres = False
    load_elite = False
    alt_cpgs = True
    add_noise = True
    generations = 10000
    max_steps = 1000
    gen_size = 10
    elite_size = 10
    mean = 0.0
    std = 0.01

    # Run neuroevolution
    neuro_evolution_train(model_type=model_type, env_type=env_type, fixed_centres=fixed_centres, generations=generations, max_steps=max_steps,
                          gen_size=gen_size, mean=mean, alt_cpgs=alt_cpgs, std=std, elite_size=elite_size, load_elite=load_elite, add_noise=add_noise)

    # RL-PIBB PARAMS
    epochs = 500
    max_steps = 1000
    rollout_size = 10
    norm_constant = 10.0
    variance = 0.05
    decay = 0.995
    test_case = 1
    alt_cpgs = False
    add_noise = True

    # Read alt_cpgs from command line
    if len(sys.argv) > 1:
        alt_cpgs = bool(sys.argv[1])
    print(f"Alt CPGs: {alt_cpgs}")
    print(f"Add noise: {add_noise}")

    # Run RL-PIBB
    rl_pibb_train(env_type, epochs, max_steps, rollout_size, norm_constant, variance, decay, alt_cpgs, add_noise, test_case)