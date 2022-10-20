#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:24:33 2020

@author: mehlc
"""

import tensorflow as tf
from tensorflow.python import ops
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import initializers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops


from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K

# =============================================================================
# CUSTOM CONSERVATION LAYERS
# =============================================================================


# CONSERVATION OF ATOMIC MASSES
class AtomicConservation(Layer):

    #@interfaces.legacy_dense_support
    def __init__(self, units,
                 param1_X,
                 param2_X,
                 param1_Y,
                 param2_Y,
                 log_transform,
                 threshold,
                 A_atomic_t,
                 A_inv_t,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.zeros(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AtomicConservation, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True
        
        self.param1_X = param1_X
        self.param2_X = param2_X
        self.param1_Y = param1_Y
        self.param2_Y = param2_Y
        self.log_transform = log_transform
        self.threshold = threshold
        self.A_atomic_t = A_atomic_t
        self.A_inv_t = A_inv_t

    
    def call(self, inputs):
        
        # Checking if we get a list of tensors
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('AtomicConservation must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        
        # Unwrapping input tensors
        prev_layer = inputs[0]
        network_input = inputs[1]
        
        # Casting atomic conservation arrays
        A_atomic_t_tens = math_ops.cast(self.A_atomic_t, dtype="float64")
        A_inv_t_tens = math_ops.cast(self.A_inv_t, dtype="float64")
        
        # Casting as float64
        param1_Y_reshaped = math_ops.cast(self.param1_Y, dtype="float64")
        param2_Y_reshaped = math_ops.cast(self.param2_Y, dtype="float64")
            
        # Casting as float64
        param1_X_reshaped = math_ops.cast(self.param1_X, dtype="float64")
        param2_X_reshaped = math_ops.cast(self.param2_X, dtype="float64")
        
        # Cast input
        prev_layer = math_ops.cast(prev_layer, self._compute_dtype)
        
        # UNSCALE PREVIOUS LAYER
        Y_new = prev_layer*tf.math.sqrt(param2_Y_reshaped) + param1_Y_reshaped


        # UNLOG PREVIOUS LAYER (only if log asked at exit - to regularize)
        if self.log_transform==1:
            Y_new = tf.math.exp(Y_new)
        elif self.log_transform==2:
            lambda_bct = 0.1
            Y_new = (lambda_bct*Y_new + 1.0)**(1./lambda_bct)
        
        # UNSCALE NETWORK INPUT
        Y_network_input = network_input[:,1:]*tf.math.sqrt(param2_X_reshaped[1:]) + param1_X_reshaped[1:]            

            
        if self.log_transform==1:
            Y_network_input = tf.math.exp(Y_network_input)
        elif self.log_transform==2:
            Y_network_input = (lambda_bct*Y_network_input + 1.0)**(1./lambda_bct)
        
        #----------------------------------------------------------------------
        # Here we manipulate species mass fraction in non-normalized space

        # Getting Yj's (atomic mass fractions) and computing missing masses
        Yj_initial = gen_math_ops.mat_mul(Y_network_input, A_atomic_t_tens)
        Yj_star = gen_math_ops.mat_mul(Y_new, A_atomic_t_tens)
        Yj_defect = Yj_initial - Yj_star

        # Computing alpha_k's (correction to balancing species: "CO2", "H2O", "O2", "N2")
        alpha_k = gen_math_ops.mat_mul(Yj_defect, A_inv_t_tens)

        # Updating Yk
        Y_corr = Y_new + alpha_k  
        outputs = Y_corr

        #----------------------------------------------------------------------
    
        # LOG OUTPUT
        if self.log_transform==1:
            outputs = tf.clip_by_value(outputs, clip_value_min=self.threshold, clip_value_max=1)
            outputs = tf.math.log(outputs)
        if self.log_transform==2:
            outputs = tf.clip_by_value(outputs, clip_value_min=0.0, clip_value_max=1)
            outputs = (outputs**lambda_bct-1.0)/lambda_bct
                
        # SCALE OUTPUT  
        outputs = (outputs - param1_Y_reshaped)/tf.math.sqrt(param2_Y_reshaped)
          

        return outputs



    def compute_output_shape(self, input_shape):
        
        assert type(input_shape) is list  # must have mutiple input shape 
        
        input_shape_prev_layer = input_shape[0]
        
        input_shape_0 = tensor_shape.TensorShape(input_shape_prev_layer)
        input_shape_0 = input_shape_0.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape_0[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape_0)
        return input_shape_0[:-1].concatenate(self.units)


    def get_config(self):
        config = {
            'units': self.units,
            'param1_X': self.param1_X,
            'param2_X': self.param2_X,
            'param1_Y': self.param1_Y,
            'param2_Y': self.param2_Y,
            'log_transform': self.log_transform,
            'threshold': self.threshold,
            'A_atomic_t': self.A_atomic_t,
            'A_inv_t': self.A_inv_t,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(AtomicConservation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# CONSERVATION OF ATOMIC MASSES WITH RR OUTPUTS
class AtomicConservation_RR(Layer):

    #@interfaces.legacy_dense_support
    def __init__(self, units,
                 param1_Y,
                 param2_Y,
                 param2_Y_scale,
                 A_atomic_t,
                 A_inv_t,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.zeros(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AtomicConservation_RR, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True
        
        self.param1_Y = param1_Y
        self.param2_Y = param2_Y
        self.param2_Y_scale = param2_Y_scale
        self.A_atomic_t = A_atomic_t
        self.A_inv_t = A_inv_t
        

    
    def call(self, inputs):    
        # Cast input
        inputs = math_ops.cast(inputs, self._compute_dtype)
        
        # Casting atomic conservation arrays
        A_atomic_t_tens = math_ops.cast(self.A_atomic_t, dtype="float64")
        A_inv_t_tens = math_ops.cast(self.A_inv_t, dtype="float64")
        
        # Casting as float64
        param1_Y_reshaped = math_ops.cast(self.param1_Y, dtype="float64")
        param2_Y_reshaped = math_ops.cast(self.param2_Y, dtype="float64")
        param2_Y_scale_reshaped = math_ops.cast(self.param2_Y_scale, dtype="float64")
        
        # UNSCALE PREVIOUS LAYER
        g = inputs*tf.math.sqrt(param2_Y_reshaped) + param1_Y_reshaped

        #----------------------------------------------------------------------
        # Here we manipulate in non-normalized space
        # Getting M*g
        Mg = gen_math_ops.mat_mul(g, A_atomic_t_tens)

        # Computing alpha_k's (correction to balancing species: "CO2", "H2O", "O2", "N2")
        alpha_k = gen_math_ops.mat_mul(Mg, A_inv_t_tens)

        # Updating Yk
        omega_k = g - alpha_k  
        outputs = omega_k
        
        #----------------------------------------------------------------------
                
        # SCALE OUTPUT
        outputs = (outputs - param1_Y_reshaped)/tf.math.sqrt(param2_Y_scale_reshaped)
        
        return outputs



    def compute_output_shape(self, input_shape):
        
        assert type(input_shape) is list  # must have mutiple input shape 
        
        input_shape_prev_layer = input_shape[0]
        
        input_shape_0 = tensor_shape.TensorShape(input_shape_prev_layer)
        input_shape_0 = input_shape_0.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape_0[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape_0)
        return input_shape_0[:-1].concatenate(self.units)


    def get_config(self):
        config = {
            'units': self.units,
            'param1_Y': self.param1_Y,
            'param2_Y': self.param2_Y,
            'param2_Y_scale': self.param2_Y_scale,
            'A_atomic_t': self.A_atomic_t,
            'A_inv_t': self.A_inv_t,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(AtomicConservation_RR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# CONSERVATION OF ATOMIC MASSES WITH RR OUTPUTS USING ORTHOGONAL PROJECTION METHOD
class AtomicConservation_RR_lsq(Layer):

    #@interfaces.legacy_dense_support
    def __init__(self, units,
                 param1_Y,
                 param2_Y,
                 param2_Y_scale,
                 L,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.zeros(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AtomicConservation_RR_lsq, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True
        
        self.param1_Y = param1_Y
        self.param2_Y = param2_Y
        self.param2_Y_scale = param2_Y_scale
        self.L = L
        

    
    def call(self, inputs):    
        # Cast input
        inputs = math_ops.cast(inputs, self._compute_dtype)

        # Casting atomic conservation arrays
        L = math_ops.cast(self.L, dtype="float64")
        
        # Casting as float64
        param1_Y_reshaped = math_ops.cast(self.param1_Y, dtype="float64")
        param2_Y_reshaped = math_ops.cast(self.param2_Y, dtype="float64")
        param2_Y_scale_reshaped = math_ops.cast(self.param2_Y_scale, dtype="float64")
        
        # UNSCALE PREVIOUS LAYER
        g = inputs*tf.math.sqrt(param2_Y_reshaped) + param1_Y_reshaped
            

        #----------------------------------------------------------------------
        # Here we manipulate in non-normalized space
        # Computing output as solution of a least square problem
        outputs = gen_math_ops.mat_mul(g, L)
        
        #----------------------------------------------------------------------
                
        # SCALE OUTPUT
        outputs = (outputs - param1_Y_reshaped)/tf.math.sqrt(param2_Y_scale_reshaped)
          
        return outputs



    def compute_output_shape(self, input_shape):
        
        assert type(input_shape) is list  # must have mutiple input shape 
        
        input_shape_prev_layer = input_shape[0]
        
        input_shape_0 = tensor_shape.TensorShape(input_shape_prev_layer)
        input_shape_0 = input_shape_0.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape_0[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape_0)
        return input_shape_0[:-1].concatenate(self.units)


    def get_config(self):
        config = {
            'units': self.units,
            'param1_Y': self.param1_Y,
            'param2_Y': self.param2_Y,
            'param2_Y_scale': self.param2_Y_scale,
            'L': self.L,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(AtomicConservation_RR_lsq, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =============================================================================
# Residual blocks layer
# =============================================================================

class ResidualBlock(Layer):
    """ A custom residual block for the ResNet Model
    """
    def __init__(self,
                 units,
                 kernel_regularizer,
                 activation,
                 kernel_initializer,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.hidden = Dense(self.units, kernel_regularizer=self.kernel_regularizer, activation=self.activation,
                                    kernel_initializer=self.kernel_initializer)
        self.outputs = Dense(self.units)
        
    def call(self, inputs):
        Z = inputs
        Z = self.hidden(Z)
        outputs = self.outputs(Z)
        if self.activation == "swish":
            return tf.keras.activations.swish(inputs + outputs)
        elif self.activation == "tanh":
            return tf.keras.activations.tanh(inputs + outputs)
        else:
            return tf.keras.activations.relu(inputs + outputs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
                       "units": self.units,
                       "kernel_regularizer": self.kernel_regularizer,
                       "activation": self.activation,
                       "kernel_initializer": self.kernel_initializer,
                       })
        return config

# =============================================================================
# CUSTOM SLICING LAYERS
# =============================================================================


class GetN2Layer(Layer):

    #@interfaces.legacy_dense_support
    def __init__(self, units,
                 n2_index,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.zeros(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GetN2Layer, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True
        
        self.n2_index = n2_index
        
    
    def call(self, inputs):    
        # Cast input
        inputs = math_ops.cast(inputs, self._compute_dtype)

        outputs = tf.reshape(inputs[:,self.n2_index], shape=[-1, 1], name="yN2")
        
        return outputs



    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.units)


    def get_config(self):
        config = {
            'units': self.units,
            'n2_index': self.n2_index,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(GetN2Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ZerosLayer(Layer):

    #@interfaces.legacy_dense_support
    def __init__(self, units,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.zeros(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ZerosLayer, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True
                
    
    def call(self, inputs):    
        # Cast input
        inputs = math_ops.cast(inputs, self._compute_dtype)

        outputs = tf.zeros(tf.shape(inputs))
        
        return outputs



    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.units)


    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(ZerosLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConstantLayer(Layer):

    #@interfaces.legacy_dense_support
    def __init__(self, units,
                 cte,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.zeros(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ConstantLayer, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True

        self.cte = cte
                
    
    def call(self, inputs):    
        # Cast input
        inputs = math_ops.cast(inputs, self._compute_dtype)

        outputs = self.cte * tf.ones(tf.shape(inputs))
        
        return outputs



    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.units)


    def get_config(self):
        config = {
            'units': self.units,
            'cte': self.cte,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(ConstantLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GetLeftPartLayer(Layer):

    #@interfaces.legacy_dense_support
    def __init__(self, units,
                 index,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.zeros(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GetLeftPartLayer, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True
        
        self.index = index
        
    
    def call(self, inputs):    
        # Cast input
        inputs = math_ops.cast(inputs, self._compute_dtype)
        
        outputs = inputs[:,:self.index]
        
        return outputs



    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.units)


    def get_config(self):
        config = {
            'units': self.units,
            'index': self.index,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(GetLeftPartLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GetRightPartLayer(Layer):

    #@interfaces.legacy_dense_support
    def __init__(self, units,
                 index,
                 kernel_initializer=initializers.GlorotNormal(),
                 bias_initializer=initializers.zeros(),
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GetRightPartLayer, self).__init__(**kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True
        
        self.index = index
        
    
    def call(self, inputs):    
        # Cast input
        inputs = math_ops.cast(inputs, self._compute_dtype)

        outputs = outputs = inputs[:, self.index:]
        
        return outputs



    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
          raise ValueError(
              'The innermost dimension of input_shape must be defined, but saw: %s'
              % input_shape)
        return input_shape[:-1].concatenate(self.units)


    def get_config(self):
        config = {
            'units': self.units,
            'index': self.index,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(GetRightPartLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# =============================================================================
# CUSTOM METRICS
# =============================================================================

# Custom metrics to check sum to 1
def sum_species_metric(param1_Y_tensor, param2_Y_tensor, log_transform):
    
    def sum_species(y_true, y_pred):

        y_true = K.cast(y_true, y_pred.dtype)


        # Reverting normalization on y
        y_tilde = K.sqrt(param2_Y_tensor)*y_pred + param1_Y_tensor
        
        # Computing sum of y's
        if log_transform==1: # Reverting log transform if needed
            y_tilde = K.exp(y_tilde)
        elif log_transform==2:
            y_tilde = (lambda_bct*y_tilde+1.0)**(1./lambda_bct)
        metric_mass = K.sum(y_tilde, axis=-1)
        
        
        return metric_mass
    
    return sum_species



def unscaled_mapes(param1_Y_tensor, param2_Y_tensor,log_transform_species,threshold,log_transform_T,targets,spec_MAPE):
    
         
        def unscaled_mape(y_true, y_pred, target):
            
            y_true = K.cast(y_true, y_pred.dtype)
            
            y_pred_unscaled = K.sqrt(param2_Y_tensor)*y_pred + param1_Y_tensor
            y_true_unscaled = K.sqrt(param2_Y_tensor)*y_true + param1_Y_tensor
                 
            if log_transform_species: 
                y_true_unscaled = K.exp(y_true_unscaled)
                y_pred_unscaled = K.exp(y_pred_unscaled)
            
            
            error=100.*(K.abs((y_true_unscaled-y_pred_unscaled)/K.maximum(y_true_unscaled,threshold)))  
            
                
            if target in targets:
                return error[:,targets.index(target)]
            elif target=='total':
                return K.mean(error,axis=1)
        
       
        metrics_list = []
        
        for spec in spec_MAPE:
            
            def f(y_true,y_pred):
                return unscaled_mape(y_true, y_pred,spec)            
            f.__name__ = "MAPE_" + spec
            
            metrics_list.append(f)

        
        def MAPE(y_true,y_pred):
            return unscaled_mape(y_true, y_pred,'total')
        metrics_list.append(MAPE)
        
        
        return metrics_list




def unscaled_maes(param1_Y_tensor, param2_Y_tensor,log_transform_species,log_transform_T,targets,spec_MAE):
    
         
        def unscaled_mae(y_true, y_pred,target):
            
            y_true = K.cast(y_true, y_pred.dtype)
            
            y_pred_unscaled = K.sqrt(param2_Y_tensor)*y_pred + param1_Y_tensor
            y_true_unscaled = K.sqrt(param2_Y_tensor)*y_true + param1_Y_tensor
                                 
            if log_transform_species: 
                y_true_unscaled = K.exp(y_true_unscaled)
                y_pred_unscaled = K.exp(y_pred_unscaled)
            
            
            error=K.abs(y_true_unscaled-y_pred_unscaled)  
            
                
            if target in targets:
                return error[:,targets.index(target)]
            elif target=='total':
                return K.mean(error,axis=1)
            
            
        # spec_MAPE = ['H2', 'H', 'CH4']
            
        metrics_list = []
        
        for spec in spec_MAE:
            
            def f(y_true,y_pred):
                return unscaled_mae(y_true, y_pred,spec)
            
            f.__name__ = "MAE_" + spec
            
            metrics_list.append(f)
            
        
        def MAE(y_true,y_pred):
            return unscaled_mae(y_true, y_pred,'total')
        metrics_list.append(MAE)

        return metrics_list


