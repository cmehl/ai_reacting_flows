import torch
import torch.nn as nn
from tensorflow import keras
from tensorflow.keras import layers

from ai_reacting_flows.ann_model_generation.Custom_layers import ResidualBlock


"""In this file we propose function to convert a MLP from pytorch to keras. This is needed to use trained networks in NNICE.
These functions have not been extended to DeepONet architectures yet """


def transfer_weights_mlp(pytorch_model, keras_model):
    """
    Transfer weights from PyTorch MLPModel to Keras MLPModel.
    
    Handles both dense layers and residual blocks automatically.
    
    Args:
        pytorch_model: PyTorch MLPModel instance
        keras_model: Keras MLPModel instance
    """
    
    # Get PyTorch layers (filter out activation functions)
    pytorch_layers = [layer for layer in pytorch_model.model 
                      if isinstance(layer, (nn.Linear, nn.Module)) 
                      and hasattr(layer, 'weight')]
    
    pytorch_layers = [layer for layer in pytorch_model.model
                      if isinstance(layer, (nn.Linear, ResidualBlock))]
    
    
    # Get Keras layers (filter out activation functions)
    keras_layers = [layer for layer in keras_model.model_layers 
                    if isinstance(layer, (layers.Dense, keras.Model, keras.layers.Layer))
                    and hasattr(layer, 'weights') and len(layer.weights) > 0]
    
    if len(pytorch_layers) != len(keras_layers):
        raise ValueError(
            f"Layer count mismatch: PyTorch has {len(pytorch_layers)} layers, "
            f"Keras has {len(keras_layers)} layers"
        )
    
    print(f"Transferring weights for {len(pytorch_layers)} layers...")
    
    for idx, (pt_layer, k_layer) in enumerate(zip(pytorch_layers, keras_layers)):
        layer_name = pt_layer.__class__.__name__
        
        # Handle standard Dense/Linear layers
        if isinstance(pt_layer, nn.Linear):
            if not isinstance(k_layer, layers.Dense):
                raise TypeError(
                    f"Layer {idx}: PyTorch Linear layer doesn't match Keras layer type "
                    f"{k_layer.__class__.__name__}"
                )
            
            weight = pt_layer.weight.detach().cpu().numpy().T  # Transpose for Keras
            bias = pt_layer.bias.detach().cpu().numpy()
            
            k_layer.set_weights([weight, bias])
            print(f"  Layer {idx}: Dense/Linear - shape {weight.shape}")
        
        # Handle ResidualBlock (custom layer)
        elif hasattr(pt_layer, '__class__') and 'Residual' in pt_layer.__class__.__name__:
            transfer_residual_block(pt_layer, k_layer, idx)
        
        else:
            print(f"  Layer {idx}: Skipping {layer_name} (no trainable weights or unsupported)")
    
    print("Weight transfer complete!")




def transfer_residual_block(pytorch_block, keras_block, layer_idx):
    """
    Transfer weights for a ResidualBlock.
    
    ResidualBlock structure:
    - linear1 + activation1
    - linear2 + activation2
    - shortcut (Linear or Identity)
    - output = activation2(linear2(activation1(linear1(x))) + shortcut(x))
    """

    print(f"  Layer {layer_idx}: ResidualBlock")
    
    # Transfer linear1
    weight1 = pytorch_block.linear1.weight.detach().cpu().numpy().T
    bias1 = pytorch_block.linear1.bias.detach().cpu().numpy()
    keras_block.linear1.set_weights([weight1, bias1])
    print(f"    linear1: shape {weight1.shape}")
    
    # Transfer linear2
    weight2 = pytorch_block.linear2.weight.detach().cpu().numpy().T
    bias2 = pytorch_block.linear2.bias.detach().cpu().numpy()
    keras_block.linear2.set_weights([weight2, bias2])
    print(f"    linear2: shape {weight2.shape}")
    
    # Transfer shortcut (only if it's a Linear layer, not Identity)
    if isinstance(pytorch_block.shortcut, nn.Linear):
        if keras_block.use_shortcut:
            weight_shortcut = pytorch_block.shortcut.weight.detach().cpu().numpy().T
            bias_shortcut = pytorch_block.shortcut.bias.detach().cpu().numpy()
            keras_block.shortcut.set_weights([weight_shortcut, bias_shortcut])
            print(f"    shortcut: shape {weight_shortcut.shape}")
        else:
            raise ValueError("PyTorch has Linear shortcut but Keras has Identity")
    else:
        if keras_block.use_shortcut:
            raise ValueError("PyTorch has Identity shortcut but Keras has Linear")
        print(f"    shortcut: Identity (no weights to transfer)")



