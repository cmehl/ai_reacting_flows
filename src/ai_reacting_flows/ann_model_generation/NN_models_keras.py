import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResidualBlock_keras(keras.layers.Layer):
    
    def __init__(self, in_features, out_features, activation, **kwargs):
        super(ResidualBlock_keras, self).__init__(**kwargs)
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear1 = layers.Dense(out_features, dtype=tf.float64, use_bias=True)
        self.activation1 = activation
        
        self.linear2 = layers.Dense(out_features, dtype=tf.float64, use_bias=True)
        self.activation2 = activation
        
        # If input and output dimensions differ, adjust with a linear layer
        if in_features != out_features:
            self.shortcut = layers.Dense(out_features, dtype=tf.float64, use_bias=True)
            self.use_shortcut = True
        else:
            self.shortcut = None
            self.use_shortcut = False
    
    def call(self, x):

        x = tf.cast(x, tf.float64)

        if self.use_shortcut:
            identity = self.shortcut(x)
        else:
            identity = x
        
        out = self.linear1(x)
        out = self.activation1(out)
        out = self.linear2(out)
        out = out + identity
        
        return self.activation2(out)
    
    def get_config(self):
        config = super(ResidualBlock_keras, self).get_config()
        config.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'activation': self.activation
        })
        return config




class MLPModel_keras(keras.Model):
    
    def __init__(self, hidden_layers, layers_type, activations, **kwargs):
        super(MLPModel_keras, self).__init__(**kwargs)
        
        self.hidden_layers = hidden_layers
        self.layers_type = layers_type
        self.activations = activations
        
        self.model_layers = []
        
        # Add hidden layers
        for i in range(1, len(hidden_layers)):
            if layers_type[i-1] == "dense":
                self.model_layers.append(
                    layers.Dense(
                        hidden_layers[i],
                        dtype=tf.float64,
                        use_bias=True
                    )
                )
                self.model_layers.append(activations[i-1])
            elif layers_type[i-1] == "resnet":
                self.model_layers.append(
                    ResidualBlock_keras(
                        hidden_layers[i-1],
                        hidden_layers[i],
                        activations[i-1]
                    )
                )
            else:
                raise ValueError(f"ERROR: layer type \"{layers_type[i-1]}\" does not exist")
    
    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
    
    def get_config(self):
        config = super(MLPModel_keras, self).get_config()
        config.update({
            'hidden_layers': self.hidden_layers,
            'layers_type': self.layers_type,
            'activations': self.activations
        })
        return config




class DeepONet_keras(keras.Model):
    
    def __init__(self, hidden_layers, layers_type, activations, n_out, n_neuron, **kwargs):
        super(DeepONet_keras, self).__init__(**kwargs)
        
        self.branch = MLPModel_keras(
            hidden_layers["branch"],
            layers_type["branch"],
            activations["branch"]
        )
        self.trunk = MLPModel_keras(
            hidden_layers["trunk"],
            layers_type["trunk"],
            activations["trunk"]
        )
        
        self.n_neurons = n_neuron
        self.n_out = n_out
        
        self.hidden_layers = hidden_layers
        self.layers_type = layers_type
        self.activations = activations
    
    def call(self, x):
        # Split input: last column is dt, rest is y
        dt = x[:, -1]
        y = x[:, :-1]
        
        # Reshape dt to (batch_size, 1)
        dt = tf.reshape(dt, (tf.shape(x)[0], 1))
        
        # Forward pass through branch and trunk
        b = self.branch(y)
        t = self.trunk(dt)
        
        # Reshape for element-wise multiplication
        batch_size = tf.shape(x)[0]
        b = tf.reshape(b, (batch_size, self.n_out, self.n_neurons))
        t = tf.reshape(t, (batch_size, self.n_out, self.n_neurons))
        
        # Element-wise multiplication and sum over neurons
        y_dt = tf.reduce_sum(b * t, axis=2)
        
        return y_dt
    
    def get_config(self):
        config = super(DeepONet_keras, self).get_config()
        config.update({
            'hidden_layers': self.hidden_layers,
            'layers_type': self.layers_type,
            'activations': self.activations,
            'n_out': self.n_out,
            'n_neuron': self.n_neurons
        })
        return config
    


class DeepONet_shift_keras(keras.Model):
    
    def __init__(self, hidden_layers, layers_type, activations, n_out, n_neuron, **kwargs):
        super(DeepONet_shift_keras, self).__init__(**kwargs)
        
        self.branch = MLPModel_keras(
            hidden_layers["branch"],
            layers_type["branch"],
            activations["branch"]
        )
        self.trunk = MLPModel_keras(
            hidden_layers["trunk"],
            layers_type["trunk"],
            activations["trunk"]
        )
        self.shift = MLPModel_keras(
            hidden_layers["shift"],
            layers_type["shift"],
            activations["shift"]
        )
        
        self.n_neurons = n_neuron
        self.n_out = n_out
        
        self.hidden_layers = hidden_layers
        self.layers_type = layers_type
        self.activations = activations
    
    def call(self, x):
        # Split input: last column is dt, rest is y
        dt = x[:, -1]
        y = x[:, :-1]
        
        # Reshape dt to (batch_size, 1)
        dt = tf.reshape(dt, (tf.shape(x)[0], 1))
        
        # Compute shift and apply to dt
        s = self.shift(y)
        dt_s = dt + s
        
        # Forward pass through branch and trunk
        b = self.branch(y)
        t = self.trunk(dt_s)
        
        # Reshape for element-wise multiplication
        batch_size = tf.shape(x)[0]
        b = tf.reshape(b, (batch_size, self.n_out, self.n_neurons))
        t = tf.reshape(t, (batch_size, self.n_out, self.n_neurons))
        
        # Element-wise multiplication and sum over neurons
        y_dt = tf.reduce_sum(b * t, axis=2)
        
        return y_dt
    
    def get_config(self):
        config = super(DeepONet_shift_keras, self).get_config()
        config.update({
            'hidden_layers': self.hidden_layers,
            'layers_type': self.layers_type,
            'activations': self.activations,
            'n_out': self.n_out,
            'n_neuron': self.n_neurons
        })
        return config


