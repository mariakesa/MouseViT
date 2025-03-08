import torch
import numpy as np
import torch
import numpy as np
from zig_model import ZIG

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