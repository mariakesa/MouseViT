import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
import torch.optim as optim
from src.zig_model import ZIG
from src.utils import evaluate_model_on_fold
import matplotlib.pyplot as plt
from torch import nn
from transformers import ViTModel, ViTImageProcessor, ViTConfig
from collections import defaultdict

# 1) Load .env
load_dotenv()

# 2) Import your pipeline steps + the allen_api
from src.data_loader import allen_api
from src.pipeline_steps import (
    AnalysisPipeline,
    AllenStimuliFetchStep,
    ImageToEmbeddingStep,
    StimulusGroupKFoldSplitterStep,
    MergeEmbeddingsStep
)

def make_container_dict(boc):
    experiment_container = boc.get_experiment_containers()
    container_ids = [dct['id'] for dct in experiment_container]
    eids = boc.get_ophys_experiments(experiment_container_ids=container_ids)
    df = pd.DataFrame(eids)
    reduced_df = df[['id', 'experiment_container_id', 'session_type']]
    grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])[
        'id'].agg(list).reset_index()
    eid_dict = {}
    for row in grouped_df.itertuples(index=False):
        c_id, sess_type, ids = row
        if c_id not in eid_dict:
            eid_dict[c_id] = {}
        eid_dict[c_id][sess_type] = ids[0]
    return eid_dict

# Reset the weights of the model to a naive stat

def main():
    # A) Allen BOC
    boc = allen_api.get_boc()

    # B) Container dict
    eid_dict = make_container_dict(boc)
    print(len(eid_dict), "containers found.")

    # C) Session->stimuli mapping
    stimulus_session_dict = {
        'three_session_A': ['natural_movie_one', 'natural_movie_three'],
        'three_session_B': ['natural_movie_one', 'natural_scenes'],
        'three_session_C': ['natural_movie_one', 'natural_movie_two'],
        'three_session_C2': ['natural_movie_one', 'natural_movie_two']
    }

    # D) HF model + processor
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    model.init_weights()
    # E) Embedding cache dir
    #embedding_cache_dir = os.environ.get('TRANSF_EMBEDDING_PATH', 'embeddings_cache')
    embedding_cache_dir='/home/maria/Documents/NullViTEmbeddings'

    # F) Build pipeline with all steps
    pipeline = AnalysisPipeline([
        AllenStimuliFetchStep(boc),
        ImageToEmbeddingStep(processor, model, embedding_cache_dir),
        StimulusGroupKFoldSplitterStep(boc, eid_dict, stimulus_session_dict, n_splits=10),
        MergeEmbeddingsStep(),  # merges the neural folds with the image embeddings
    ])

    # G) Run pipeline on a single container/session/stimulus
    container_id = 511498742
    #session = 'three_session_A'
    #stimulus = 'natural_movie_three'
    session='three_session_B'
    stimulus='natural_scenes'
    result = pipeline.run((container_id, session, stimulus))

    # H) Print final results
    print("\n=== FINAL PIPELINE OUTPUT ===")
    print("Keys in 'result':", list(result.keys()))
    #  'raw_data_dct', 'embedding_file', 'folds', 'merged_folds', etc.

    print(f"Embedding file path: {result['embedding_file']}")
    folds = result['folds']
    print(f"Number of folds: {len(folds)}")

    merged_folds = result['merged_folds']
    for i, fold_data in enumerate(merged_folds, start=1):
        (Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test) = fold_data
        print(f"\nFold {i}:")
        print(f"  Xn_train: {Xn_train.shape}, Xe_train: {Xe_train.shape}")
        print(f"  Xn_test : {Xn_test.shape},  Xe_test : {Xe_test.shape}")
        print(f"  frames_train: {frames_train.shape}, frames_test: {frames_test.shape}")

    return merged_folds
'''
def regression(merged_folds, fold=0, save_path="trained_models/zig_model_fold.pth"):
    """
    Train a ZIG model on a specific fold and save the trained model.

    Args:
        merged_folds (list): List of folds containing (Xn_train, Xe_train, _, _, _, _).
        fold (int): Index of the fold to use for training. Defaults to 0.
        save_path (str): Path to save the trained model.
    """

    # Validate fold index
    if fold < 0 or fold >= len(merged_folds):
        raise ValueError(f"Invalid fold index {fold}. Must be between 0 and {len(merged_folds) - 1}.")

    # Extract a single fold
    Xn_train, Xe_train, _, _, _, _ = merged_folds[fold]

    # Convert to tensors
    Xn_train_tensor = torch.tensor(Xn_train, dtype=torch.float32)
    Xe_train_tensor = torch.tensor(Xe_train, dtype=torch.float32)

    # Model hyperparameters
    yDim = Xn_train_tensor.shape[1]  # Output dimensions (embedding size)
    xDim = Xe_train_tensor.shape[1]  # Input dimensions (neural activity size)
    gen_nodes = 128  # Number of hidden layer neurons
    factor = torch.min(Xn_train_tensor, axis=0).values  # Default factor for initialization

    # Initialize ZIG model
    model = ZIG(yDim, xDim, gen_nodes, factor)

    model.train()
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    batch_size = 32
    num_batches = len(Xe_train_tensor) // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_neurons = 0  # Keep track of the total number of neurons
        total_data_points =0
        for i in range(num_batches):
            # Batch slicing
            start = i * batch_size
            end = start + batch_size
            X_batch = Xe_train_tensor[start:end]
            Y_batch = Xn_train_tensor[start:end]
            n_time = Y_batch.shape[0]
            #print(Y_batch.shape)

            # Forward pass
            loss, _, _, _, _, _ = model(X_batch, Y_batch)

            loss = -loss  # Negative loss because entropy loss is a sum

            # Get number of neurons in the batch
            num_neurons = Y_batch.shape[1]  # Assuming second dimension is neurons

            # Accumulate total loss and neuron count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            #total_neurons += num_neurons
            total_data_points += n_time * num_neurons

        avg_surprise_per_data_point = epoch_loss / total_data_points
        avg_p_per_data_point = np.exp(-avg_surprise_per_data_point)

        # Compute average surprise per neuron
        #avg_surprise_per_neuron = epoch_loss / total_neurons if total_neurons > 0 else 0
        #avg_p_per_neuron = np.exp(-avg_surprise_per_neuron)  # Compute probability from surprise

        # Print epoch progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}, "
            f"Avg Surprise per Data Point: {avg_surprise_per_data_point:.4f}, "
            f"Avg p per Data Point: {avg_p_per_data_point:.4f}")

    print(f"Training complete on fold {fold}.")

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    test_probs=evaluate_model_on_fold(merged_folds, fold, model_path=save_path)
    return test_probs

def regression(merged_folds, fold=0, save_path="trained_models/zig_model_fold.pth"):
    """
    Train a ZIG model on a specific fold and save the trained model.

    Args:
        merged_folds (list): List of folds containing (Xn_train, Xe_train, _, _, _, _).
        fold (int): Index of the fold to use for training. Defaults to 0.
        save_path (str): Path to save the trained model.
    """

    # Validate fold index
    if fold < 0 or fold >= len(merged_folds):
        raise ValueError(f"Invalid fold index {fold}. Must be between 0 and {len(merged_folds) - 1}.")

    # Extract a single fold
    Xn_train, Xe_train, _, _, _, _ = merged_folds[fold]

    # Convert to tensors
    Xn_train_tensor = torch.tensor(Xn_train, dtype=torch.float32)
    Xe_train_tensor = torch.tensor(Xe_train, dtype=torch.float32)

    # Model hyperparameters
    yDim = Xn_train_tensor.shape[1]  # Output dimensions (embedding size)
    xDim = Xe_train_tensor.shape[1]  # Input dimensions (neural activity size)
    gen_nodes = 128  # Number of hidden layer neurons
    factor = torch.min(Xn_train_tensor, axis=0).values  # Default factor for initialization

    # Initialize ZIG model
    model = ZIG(yDim, xDim, gen_nodes, factor)

    model.train()
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    batch_size = 32
    num_batches = len(Xe_train_tensor) // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        total_neurons = 0  # Keep track of the total number of neurons
        total_data_points =0
        for i in range(num_batches):
            # Batch slicing
            start = i * batch_size
            end = start + batch_size
            X_batch = Xe_train_tensor[start:end]
            Y_batch = Xn_train_tensor[start:end]
            n_time = Y_batch.shape[0]
            #print(Y_batch.shape)

            # Forward pass
            loss, p = model(X_batch, Y_batch)

            loss = -loss  # Negative loss because entropy loss is a sum

            # Get number of neurons in the batch
            num_neurons = Y_batch.shape[1]  # Assuming second dimension is neurons

            # Accumulate total loss and neuron count
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            #total_neurons += num_neurons
            total_data_points += n_time * num_neurons

        avg_surprise_per_data_point = epoch_loss / total_data_points
        avg_p_per_data_point = np.exp(-avg_surprise_per_data_point)

        # Compute average surprise per neuron
        #avg_surprise_per_neuron = epoch_loss / total_neurons if total_neurons > 0 else 0
        #avg_p_per_neuron = np.exp(-avg_surprise_per_neuron)  # Compute probability from surprise

        # Print epoch progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/num_batches:.4f}, "
            f"Avg Surprise per Data Point: {avg_surprise_per_data_point:.4f}, "
            f"Avg p per Data Point: {avg_p_per_data_point:.4f}")

    print(f"Training complete on fold {fold}.")

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    test_probs=evaluate_model_on_fold(merged_folds, fold, model_path=save_path)
    return test_probs
'''
import torch
import torch.optim as optim
import numpy as np

# If needed, import your ZIGBinaryEvent model.
# from your_module import ZIGBinaryEvent

import torch
import torch.optim as optim
import numpy as np
import math

import torch
import torch.optim as optim
import numpy as np
import math

def regression(merged_folds, fold=0, save_path="trained_models/zig_binary_event_fold.pth"):
    """
    Train a binary-event (focal loss) model on a specific fold, logging focal loss,
    average surprise for the correct label, probability assigned to the correct label,
    AND average probability the model assigns to actual events.

    Args:
        merged_folds (list): List of folds containing (Xn_train, Xe_train, _, _, _, _).
        fold (int): Index of the fold to use for training.
        save_path (str): Path to save the trained model state dictionary.
    """
    # Validate fold index
    if fold < 0 or fold >= len(merged_folds):
        raise ValueError(f"Invalid fold index {fold}. Must be between 0 and {len(merged_folds) - 1}.")

    # --- 1) Extract training data for this fold ---
    Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test= merged_folds[fold]

    # Convert to tensors
    Xn_train_tensor = torch.tensor(Xn_train, dtype=torch.float32)
    Xe_train_tensor = torch.tensor(Xe_train, dtype=torch.float32)

    frame_to_indices = defaultdict(list)
    for idx, frame in enumerate(frames_train):
        frame_to_indices[frame].append(idx)

    unique_frames = sorted(frame_to_indices.keys())

    Xn_avg = np.array([
        Xn_train[frame_to_indices[frame]].mean(axis=0)
        for frame in unique_frames
    ])

    Xe_first = np.array([
        Xe_train[frame_to_indices[frame][0]]
        for frame in unique_frames
    ])
    Xn_norm = (Xn_avg - Xn_avg.min(axis=0)) / (Xn_avg.max(axis=0) - Xn_avg.min(axis=0))

    Xn_train_tensor = torch.tensor(Xn_norm)
    Xe_train_tensor = torch.tensor(Xe_first)

    # --- 2) Model hyperparameters ---
    yDim = Xn_train_tensor.shape[1]  # Number of neurons (outputs)
    xDim = Xe_train_tensor.shape[1]  # Embedding dim
    gen_nodes = 128
    alpha = 0.75  # Weight for positive (event) examples
    gamma = 5.0   # Focal exponent
    p_dropout = 0.5

    # --- 3) Initialize the binary-event model (focal loss) ---
    # Make sure the import path is correct for your setup.
    model = ZIG(
        neuronDim=yDim,
        xDim=xDim
    )

    # --- 4) Set up optimizer ---
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # --- 5) Training loop ---
    num_epochs = 100
    batch_size = 32
    num_batches = max(len(Xe_train_tensor) // batch_size, 1)

    for epoch in range(num_epochs):
        epoch_focal_loss = 0.0

        epoch_surprise_sum = 0.0         # Sum of -log(correct_prob)
        total_data_points = 0

        sum_event_prob = 0.0            # Sum of p for actual-event samples
        sum_event_count = 0             # Count of actual-event samples

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size

            X_batch = Xe_train_tensor[start:end]
            Y_batch = Xn_train_tensor[start:end]

            # 1) Forward pass => (focal_loss, p)
            focal_loss, p = model(X_batch, Y_batch)

            # 2) Backprop & update
            optimizer.zero_grad()
            focal_loss = focal_loss.mean()
            focal_loss.backward()
            #focal_loss.backward()
            optimizer.step()

            # 3) Accumulate focal loss
            epoch_focal_loss += focal_loss.item()

            # --- 4) Calculate surprise & event probabilities ---
            with torch.no_grad():
                # Probability assigned to the correct label
                correct_prob = torch.where(Y_batch > 0, p, 1 - p)
                eps = 1e-8
                correct_prob = torch.clamp(correct_prob, min=eps)
                surprise = -torch.log(correct_prob)  # negative log-likelihood of correct label

                epoch_surprise_sum += surprise.sum().item()
                total_data_points += correct_prob.numel()

                # Probability that the model assigns to actual events only
                event_mask = (Y_batch > 0)
                if event_mask.any():
                    # Sum up all p for those event entries
                    sum_event_prob += p[event_mask].sum().item()
                    sum_event_count += event_mask.sum().item()

        # --- 5) Aggregate metrics for this epoch ---
        avg_focal_loss = epoch_focal_loss / num_batches
        avg_surprise = epoch_surprise_sum / total_data_points if total_data_points > 0 else 0
        avg_correct_label_prob = math.exp(-avg_surprise) if avg_surprise > 0 else 1.0

        # Probability assigned to actual events
        if sum_event_count > 0:
            avg_prob_for_events = sum_event_prob / sum_event_count
        else:
            avg_prob_for_events = 0.0

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"| Focal Loss: {avg_focal_loss:.4f} "
            f"| Avg Surprise: {avg_surprise:.4f} "
            f"| Prob(Correct Label): {avg_correct_label_prob:.4f} "
            f"| Avg Prob(Event=1 for Y=1): {avg_prob_for_events:.4f}"
        )

    # --- 6) Save trained model ---
    torch.save(model.state_dict(), save_path)
    print(f"Training complete on fold {fold}. Model saved to {save_path}")
    #evaluate_model_on_fold(merged_folds, fold, model_path=save_path)
    return evaluate_model_on_fold(merged_folds, fold, model_path=save_path)







if __name__ == "__main__":
    merged_dat=main()
    test_probs, event_probs=regression(merged_dat)
    test_probs=np.array(test_probs)
    event_probs=np.array(event_probs)
    np.save('test_probs_.npy', test_probs)
    np.save('event_probs_.npy', event_probs)
    print(test_probs)
    print(event_probs)
    #plt.hist(test_probs, bins=50)
    #plt.show()