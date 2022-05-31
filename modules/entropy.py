""" Computes entropy density for different Ising models. """
import numpy as np
import subprocess


def get_entropy(dimension: int, interaction_K: list):
    interaction_dist = len(interaction_K)
    kn_interactions = interaction_K[1:]
    kn_interactions_bool_list = [b == 0 for b in kn_interactions]
    nearest_neighbor_bool = interaction_dist == 1 or (interaction_dist > 1 and all(kn_interactions_bool_list))
    k2_bool = (interaction_K[1] != 0) and (interaction_K[2] == 0)
    k3_bool = (interaction_K[1] != 0) and (interaction_K[2] != 0)

    if dimension == 1:
        if nearest_neighbor_bool:
            interaction_K = interaction_K[0]
            entropy = np.log(2) + np.log(np.cosh(interaction_K)) - interaction_K * np.tanh(interaction_K)
            return entropy
        else:
            k1 = interaction_K[0]
            k2 = interaction_K[1]
            k3 = interaction_K[2]
            if k2_bool:
                wolframscript_path = "/System/Volumes/Data/Users/jesselin/Dropbox/src/thesis/modules/mathematica/1DNNN.wls"
                completed_process = subprocess.run(["wolframscript", wolframscript_path, str(k1), str(k2)],
                                               capture_output=True, text=True)
                entropy = completed_process.stdout
                entropy = entropy.strip() # removes \n
                entropy = entropy.replace("*^", "e") # converts Mathematica to Python scientific notation
                return entropy
            elif k3_bool:
                wolframscript_path = "/System/Volumes/Data/Users/jesselin/Dropbox/src/thesis/modules/mathematica/1DNNNN.wls"
                completed_process = subprocess.run(["wolframscript", wolframscript_path, str(k1), str(k2), str(k3)],
                                                   capture_output=True, text=True)
                entropy = completed_process.stdout
                entropy = entropy.strip() # removes \n
                entropy = entropy.replace("*^", "e") # converts Mathematica to Python scientific notation
                return entropy

    elif dimension == 2:
        # check for NN
        if nearest_neighbor_bool:
            interaction_K = interaction_K[0]
            # hardcoded bc lazy
            wolframscript_path = "/System/Volumes/Data/Users/jesselin/Dropbox/src/thesis/modules/mathematica/2Dentropy.wls"
            completed_process = subprocess.run(["wolframscript", wolframscript_path, str(interaction_K)],
                                               capture_output=True, text=True)
            entropy = completed_process.stdout
            entropy = entropy.strip() # removes \n
            entropy = entropy.replace("*^", "e") # converts Mathematica to Python scientific notation
            return float(entropy)

        else:
            raise Exception("2D case must have NN interactions.")
    else:
        raise Exception("Only have dimensions <3.")
