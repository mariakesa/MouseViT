import pandas as pd
import numpy as np
import torch
from src.zig_model import ZIG
import torch
import numpy as np

def make_container_dict(boc):
    '''
    Parses which experimental id's (values)
    correspond to which experiment containers (keys).
    '''
    experiment_container = boc.get_experiment_containers()
    container_ids = [dct['id'] for dct in experiment_container]
    eids = boc.get_ophys_experiments(experiment_container_ids=container_ids)
    df = pd.DataFrame(eids)
    reduced_df = df[['id', 'experiment_container_id', 'session_type']]
    grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])[
        'id'].agg(list).reset_index()
    eid_dict = {}
    for row in grouped_df.itertuples(index=False):
        container_id, session_type, ids = row
        if container_id not in eid_dict:
            eid_dict[container_id] = {}
        eid_dict[container_id][session_type] = ids[0]
    return eid_dict


def evaluate_model_on_fold(merged_folds, fold, model_path="/home/maria/MouseViT/trained_models/zig_model_fold.pth", save_path=None):
    """
    Evaluates the trained ZIG model on a specific test fold and computes the likelihood of the observed data 
    under the model.

    Args:
        merged_folds (list): List of folds containing (Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test).
        fold (int): The fold number to evaluate.
        model_path (str): Path to the trained model checkpoint.
        save_path (str, optional): If provided, saves the test probabilities as a .npy file.

    Returns:
        test_likelihoods (numpy.ndarray): A 2D NumPy array (num_neurons, num_time_points) containing the likelihood of each test data point.
    """

    # Load the trained model
    checkpoint = torch.load(model_path, map_location="cpu")  # Load model to CPU
    Xn_train, Xe_train, _, _, _, _ = merged_folds[fold]
    
    yDim = Xn_train.shape[1]  # Number of neurons (output dimension)
    xDim = Xe_train.shape[1]  # Input dimension (ViT embeddings)
    
    # Initialize model and load weights
    gen_nodes = 128  # Keep this consistent with training
    factor = np.min(Xn_train, axis=0) 
    model = ZIG(yDim, xDim, gen_nodes, factor)
    model.load_state_dict(checkpoint)
    model.eval()  # Set model to evaluation mode

    # Select the fold's test data
    _, _, Xn_test, Xe_test, _, _ = merged_folds[fold]

    # Convert test data to PyTorch tensors
    Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
    Xe_test_tensor = torch.tensor(Xe_test, dtype=torch.float32)

    # Forward pass (no gradients needed since we are evaluating)
    with torch.no_grad():
        theta, k, p, loc, rate = model(Xe_test_tensor)  # Get model outputs

    # Compute probability of the observed data (Xn_test) under the ZIG model
    eps = 1e-6  # Small value for numerical stability
    mask = (Xn_test_tensor != 0)  # Identify nonzero spikes

    # Compute the probability of observed spike counts using the ZIG model:
    p_zeros = 1 - p  # Probability of being in the zero-inflated state
    p_spike = p * torch.exp(-k * torch.log(theta) - (Xn_test_tensor - loc) / theta) * \
              torch.exp((k - 1) * torch.log(torch.clamp(Xn_test_tensor - loc, min=eps)) - torch.lgamma(k))

    # Use mask to apply zero-inflation correctly:
    test_likelihoods = torch.where(mask, p_spike, p_zeros + eps).cpu().numpy()

    # Save as a .npy file if a save path is provided
    if save_path:
        np.save(save_path, test_likelihoods)
        print(f"Saved test likelihoods to {save_path}")

    print(f"Evaluated fold {fold}. Test likelihoods array shape: {test_likelihoods.shape}")

    return test_likelihoods  # Return full 2D array (num_neurons, num_time_points)

'''
def evaluate_model_on_fold(merged_folds, fold, model_path="/home/maria/MouseViT/trained_models/zig_model_fold.pth", save_path=None):
    """
    Evaluates the trained ZIG model on a specific test fold and stores per-time-point probabilities
    in a NumPy array where each row corresponds to a neuron and each column corresponds to a time point.

    Args:
        merged_folds (list): List of folds containing (Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test).
        fold (int): The fold number to evaluate.
        model_path (str): Path to the trained model checkpoint.
        save_path (str, optional): If provided, saves the test probabilities as a .npy file.

    Returns:
        test_prob_array (numpy.ndarray): A 2D NumPy array of shape (num_neurons, num_time_points).
    """

    # Load the trained model
    checkpoint = torch.load(model_path, map_location="cpu")  # Load model to CPU
    Xn_train, Xe_train, _, _, _, _ = merged_folds[fold]
    
    yDim = Xn_train.shape[1]  # Number of neurons (output dimension)
    xDim = Xe_train.shape[1]  # Input dimension (ViT embeddings)
    
    # Initialize model and load weights
    gen_nodes = 128  # Keep this consistent with training
    factor = np.min(Xn_train, axis=0) 
    model = ZIG(yDim, xDim, gen_nodes, factor)
    model.load_state_dict(checkpoint)
    model.eval()  # Set model to evaluation mode

    # Select the fold's test data
    _, _, Xn_test, Xe_test, _, _ = merged_folds[fold]

    # Convert test data to PyTorch tensors
    Xn_test_tensor = torch.tensor(Xn_test, dtype=torch.float32)
    Xe_test_tensor = torch.tensor(Xe_test, dtype=torch.float32)

    # Forward pass (no gradients needed since we are evaluating)
    with torch.no_grad():
        loss, theta, k, p, loc, rate = model(Xe_test_tensor, Xn_test_tensor)  # Get model outputs

    # Compute per-time-point probabilities (avoid exponentiating total loss!)
    test_prob_array = (rate.cpu().numpy())  # Directly use model output rates

    # Ensure valid probability values in (0,1]
    #test_prob_array = np.clip(test_prob_array, 1e-6, 1.0)

    # Save as a .npy file if a save path is provided
    if save_path:
        np.save(save_path, test_prob_array)
        print(f"Saved test probabilities to {save_path}")

    print(f"Evaluated fold {fold}. Test probability array shape: {test_prob_array.shape}")

    return test_prob_array  # Return full 2D array (num_neurons, num_time_points)
'''