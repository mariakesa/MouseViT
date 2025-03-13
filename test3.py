import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import torch
import torch.optim as optim
from src.zig_model import ZIG
from src.utils import evaluate_model_on_fold
import matplotlib.pyplot as plt

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
from transformers import ViTModel, ViTImageProcessor

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

    # E) Embedding cache dir
    embedding_cache_dir = os.environ.get('TRANSF_EMBEDDING_PATH', 'embeddings_cache')

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






if __name__ == "__main__":
    merged_dat=main()
    test_probs=regression(merged_dat)
    test_probs=np.array(test_probs)
    np.save('test_probs.npy', test_probs)
    print(test_probs)
    #plt.hist(test_probs, bins=50)
    #plt.show()