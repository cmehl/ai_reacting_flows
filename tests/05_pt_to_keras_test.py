import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: quieter CPU logs

import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from ai_reacting_flows.ann_model_generation.Custom_layers import ResidualBlock
from ai_reacting_flows.ann_model_generation.NN_models_keras import ResidualBlock_keras, MLPModel_keras
from ai_reacting_flows.ann_model_generation.NN_models import MLPModel
from ai_reacting_flows.ann_model_generation.NN_utilities import transfer_weights_mlp, transfer_residual_block



def compare_models_predictions(pytorch_model, keras_model, input_data):
    """
    Verify that weight transfer was successful by comparing outputs.
    
    Args:
        pytorch_model: PyTorch model
        keras_model: Keras model
        input_data: numpy array of input data
        
    Returns:
        dict with comparison metrics
    """
    # PyTorch forward pass
    pytorch_model.eval()
    with torch.no_grad():
        input_torch = torch.from_numpy(input_data).to(
            next(pytorch_model.parameters()).device
        )
        output_pytorch = pytorch_model(input_torch).cpu().numpy()
    
    # Keras forward pass
    output_keras = keras_model(input_data).numpy()
    
    # Compute differences
    difference = np.abs(output_pytorch - output_keras)
    max_diff = np.max(difference)
    matches = np.allclose(output_pytorch, output_keras, atol=1e-5)
    
    print(f"   Maximum difference: {max_diff:.2e}")
    print(f"   Outputs match (atol=1e-5): {matches}")
    print("")

    return matches



# Example usage
def test_pt_to_keras():

    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)

    print("="*70)
    print("TEST PYTORCH TO KERAS WEIGHT TRANSFER")
    print("="*70)
    
    print("\n--- TEST 1: ResidualBlock with in_features=out_features ---")
    in_features = 32
    out_features = 32
    
    pt_resblock = ResidualBlock(in_features, out_features, nn.ReLU)
    k_resblock = ResidualBlock_keras(in_features, out_features, layers.Activation('relu'))
    
    # Build Keras layer
    dummy = np.random.randn(1, in_features).astype(np.float64)
    _ = k_resblock(dummy)
    
    # Transfer weights
    transfer_residual_block(pt_resblock, k_resblock, 0)
    
    # Test forward pass
    test_input = np.random.randn(5, in_features).astype(np.float64)
    
    pt_resblock.eval()
    with torch.no_grad():
        pt_output = pt_resblock(torch.from_numpy(test_input)).numpy()
    
    k_output = k_resblock(test_input).numpy()
    
    matches = np.allclose(pt_output, k_output, atol=1e-5)

    print(f"\nOutput comparison:")
    print(f"  Max difference: {np.max(np.abs(pt_output - k_output)):.2e}")
    print(f"  Outputs match: {matches}")    
    assert(matches)

    print("\n--- TEST 2: ResidualBlock with in_features!=out_features ---")

    in_features = 32
    out_features = 64
    
    pt_resblock2 = ResidualBlock(in_features, out_features, nn.ReLU)
    k_resblock2 = ResidualBlock_keras(in_features, out_features, layers.Activation('relu'))
    
    # Build Keras layer
    dummy = np.random.randn(1, in_features).astype(np.float64)
    _ = k_resblock2(dummy)
    
    # Transfer weights
    transfer_residual_block(pt_resblock2, k_resblock2, 1)
    
    # Test forward pass
    test_input = np.random.randn(5, in_features).astype(np.float64)
    
    pt_resblock2.eval()
    with torch.no_grad():
        pt_output2 = pt_resblock2(torch.from_numpy(test_input)).numpy()
    
    k_output2 = k_resblock2(test_input).numpy()

    matches = np.allclose(pt_output, k_output, atol=1e-5)
    
    print(f"\nOutput comparison:")
    print(f"  Max difference: {np.max(np.abs(pt_output2 - k_output2)):.2e}")
    print(f"  Outputs match: {matches}")
    assert(matches)

    
    print("\n--- TEST 3: full MLP model with dense layers ---\n")

    # Define architecture
    hidden_layers = [10, 64, 32, 2]
    layers_type = ["dense", "dense", "dense"]
    activations = [nn.ReLU, nn.ReLU, nn.Identity]  # PyTorch activations
    activations_keras = [
        layers.Activation('relu'),
        layers.Activation('relu'),
        layers.Activation('linear')
    ]

    # Create PyTorch model
    device = torch.device('cpu')
    pytorch_model = MLPModel(device, hidden_layers, layers_type, activations)
    # print(pytorch_model)

    # Create Keras model
    keras_model = MLPModel_keras(hidden_layers, layers_type, activations_keras)
    # Build Keras model
    dummy_input = np.random.randn(1, hidden_layers[0]).astype(np.float64)
    _ = keras_model(dummy_input)  # Build the model
    # keras_model.summary()

    # Input on which to test
    test_input = np.ones((5, hidden_layers[0])).astype(np.float64)

    # Check that prediction for both network is initially different
    # print("\n>>> Models inference comparison before transfer:")
    # matches = compare_models_predictions(pytorch_model, keras_model, test_input)

    # # Transfer weights
    transfer_weights_mlp(pytorch_model, keras_model)

    # Verify transfer
    print("\n>>> Models inference comparison after transfer:")
    matches = compare_models_predictions(pytorch_model, keras_model, test_input)
    assert(matches)


    print("\n--- TEST 4: full MLP model with resblock layers ---\¬")

    # Define architecture
    hidden_layers = [10, 64, 32, 2]
    layers_type = ["resnet", "resnet", "resnet"]
    activations = [nn.ReLU, nn.ReLU, nn.Identity]  # PyTorch activations
    activations_keras = [
        layers.Activation('relu'),
        layers.Activation('relu'),
        layers.Activation('linear')
    ]

    # Create PyTorch model
    device = torch.device('cpu')
    pytorch_model = MLPModel(device, hidden_layers, layers_type, activations)
    # print(pytorch_model)

    # Create Keras model
    keras_model = MLPModel_keras(hidden_layers, layers_type, activations_keras)
    # Build Keras model
    dummy_input = np.random.randn(1, hidden_layers[0]).astype(np.float64)
    _ = keras_model(dummy_input)  # Build the model
    # keras_model.summary()

    # Input on which to test
    test_input = np.ones((5, hidden_layers[0])).astype(np.float64)

    # Check that prediction for both network is initially different
    # print("\n>>> Models inference comparison before transfer:")
    # matches = compare_models_predictions(pytorch_model, keras_model, test_input)

    # # Transfer weights
    transfer_weights_mlp(pytorch_model, keras_model)

    # Verify transfer
    print("\n>>> Models inference comparison after transfer:")
    matches = compare_models_predictions(pytorch_model, keras_model, test_input)
    assert(matches)