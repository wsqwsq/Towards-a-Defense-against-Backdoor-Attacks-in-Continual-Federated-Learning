"""
Robust secure aggregation oracle based on the following paper:

Robust aggregation for federated learning.
    Krishna Pillutla, Sham M Kakade, and Zaid Harchaoui.
    https://arxiv.org/abs/1912.13445
"""


import tensorflow as tf 
import tensorflow_federated as tff 



def get_median(value, weight, round_model_delta):
        
        grad_type = round_model_delta.type_signature.member
        value_type = value.type_signature.member
        
        @tff.tf_computation(grad_type, value_type, tf.float32) 
        @tf.function 
        def weiszfeld(tf_gradient, tf_updates, tf_weight): 
                diff = []
                update = tf.nest.flatten(tf_updates)
                grad = tf.nest.flatten(tf_gradient)
                for j in range(len(update)):
                    diff.append(update[j]-grad[j])
                norm = tf.linalg.global_norm(diff)
                below = tf.cast(1e-5, tf.float32)
                if norm > below:
                    below = norm
                weight = tf.math.divide_no_nan(tf_weight, below)
                return weight
        
        return tff.federated_map(weiszfeld, 
                                 (tff.federated_broadcast(round_model_delta), 
                                  value, weight))