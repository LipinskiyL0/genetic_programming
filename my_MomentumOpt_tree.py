'''
2023-04-12 Строим метод моментной оптимизации под keras, TF с вычислением момента через 
дерево
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow import keras

class MyMomentumOptimizer_tree(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, tree=None,name="MyMomentumOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate
        #self._set_hyper("decay", self._initial_decay) # 
        self._set_hyper("momentum", momentum)
        self.tree=tree
    
    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "momentum")
        
        for var in var_list:
            self.add_slot(var, "slot_s")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """

        
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
        momentum_var = self.get_slot(var, "momentum")
        slot_s_var = self.get_slot(var, "slot_s")
        momentum_hyper = self._get_hyper("momentum", var_dtype)
        params={'m0':momentum_var, 'b':momentum_hyper,  'grad':grad, 's0':slot_s_var, 's1':slot_s_var}
        moment=self.tree.eval(params)
        # momentum_var.assign(momentum_var * momentum_hyper - (1. - momentum_hyper)* grad)
        # var.assign_add(momentum_var * lr_t)
        momentum_var.assign(moment)
        slot_s_var.assign(params['s1'])
        var.assign_add(moment * lr_t)
        
       

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            #"decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }