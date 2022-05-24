"""Implement and simulate backdoor attacks in federated learning systems."""

import attr 
import tensorflow as tf 
import tensorflow_federated as tff 
from tensorflow_federated.python.tensorflow_libs import tensor_utils
from fedaggregate_RFA import get_median



def build_stateless_mean(): 
  """tff.federated_mean with empty state.""" 
  return tff.utils.StatefulAggregateFn( 
      initialize_fn=lambda: (), 
      next_fn=lambda state, value, weight=None: (
          state, tff.federated_mean(value, weight=weight))) 

    
@attr.s(eq=False, frozen=True) 
class ClientOutput(object): 
  """Structure for outputs returned from clients during federated optimization.""" 
  
  weights_delta = attr.ib() 
  weights_delta_weight = attr.ib() 
  model_output = attr.ib() 
  optimizer_output = attr.ib() 
  gradients_eg = attr.ib() 
  rep = attr.ib()
  malicious = attr.ib()
  weights_noise = attr.ib()


@attr.s(eq=False, frozen=True) 
class ServerState(object): 
  """Structure for state on the server.""" 

  model = attr.ib() 
  optimizer_state = attr.ib() 
  delta_aggregate_state = attr.ib() 


def _create_optimizer_vars(model, optimizer): 
  model_weights = _get_weights(model) 
  delta = tf.nest.map_structure(tf.zeros_like, model_weights.trainable) 
  grads_and_vars = tf.nest.map_structure( 
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(delta), 
      tf.nest.flatten(model_weights.trainable)) 
  optimizer.apply_gradients(grads_and_vars, name='server_update') 
  return optimizer.variables() 


def _get_weights(model): 
  return tff.learning.framework.ModelWeights.from_model(model) 


def _get_norm(weights): 
    return tf.linalg.global_norm(tf.nest.flatten(weights)) 


@tf.function 
def server_update(model, server_optimizer, server_optimizer_vars, server_state, 
                  weights_delta, new_delta_aggregate_state): 
  """Updates `server_state` based on `weights_delta`.""" 

  model_weights = _get_weights(model) 
  tf.nest.map_structure(lambda a, b: a.assign(b), 
                        (model_weights, server_optimizer_vars), 
                        (server_state.model, server_state.optimizer_state)) 

  grads_and_vars = tf.nest.map_structure( 
      lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta), 
      tf.nest.flatten(model_weights.trainable)) 
  server_optimizer.apply_gradients(grads_and_vars, name='server_update') 

  return tff.utils.update_state( 
      server_state, 
      model=model_weights, 
      optimizer_state=server_optimizer_vars, 
      delta_aggregate_state=new_delta_aggregate_state) 


def build_server_init_fn(model_fn, server_optimizer_fn, delta_aggregate_state): 
  """Builds a `tff.tf_computation` that returns initial `ServerState`.""" 

  @tff.tf_computation 
  def server_init_tf(): 
    model = model_fn() 
    server_optimizer = server_optimizer_fn() 
    server_optimizer_vars = _create_optimizer_vars(model, server_optimizer) 
    return ServerState( 
        model=_get_weights(model), 
        optimizer_state=server_optimizer_vars, 
        delta_aggregate_state=delta_aggregate_state) 

  return server_init_tf 


def build_server_update_fn(model_fn, server_optimizer_fn, server_state_type, 
                           model_weights_type): 
  """Builds a `tff.tf_computation` that updates `ServerState`.""" 

  @tff.tf_computation(server_state_type, model_weights_type.trainable, 
                      server_state_type.delta_aggregate_state) 
  def server_update_tf(server_state, model_delta, new_delta_aggregate_state): 
    """Updates the `server_state`."""

    model = model_fn() 
    server_optimizer = server_optimizer_fn() 
    server_optimizer_vars = _create_optimizer_vars(model, server_optimizer) 

    return server_update(model, server_optimizer, server_optimizer_vars, 
                         server_state, model_delta, new_delta_aggregate_state) 

  return server_update_tf 


def build_client_update_fn(model_fn, optimizer_fn, client_update_tf, 
                           tf_dataset_type, model_weights_type): 
  """Builds a `tff.tf_computation` in the presense of malicious clients.""" 

  @tff.tf_computation(tf_dataset_type, tf.bool, 
                      model_weights_type)
  def client_delta_tf(benign_dataset, client_type, 
                      initial_model_weights): 
    """Performs client local model optimization."""

    model = model_fn() 
    optimizer = optimizer_fn() 
    return client_update_tf(model, optimizer, benign_dataset, 
                            client_type, initial_model_weights) 

  return client_delta_tf 


def build_run_one_round_fn_attacked(server_update_fn, client_update_fn, 
                                    dummy_model_for_metadata, 
                                    federated_server_state_type, 
                                    federated_dataset_type): 
  """Builds a `tff.federated_computation` for a round of training.""" 
  
  federated_bool_type = tff.FederatedType(tf.bool, tff.CLIENTS) 

  @tff.federated_computation(federated_server_state_type, 
                             federated_dataset_type,
                             federated_bool_type)
  def run_one_round(server_state, federated_dataset, 
                    malicious_clients): 
    """Orchestration logic for one round of computation.""" 

    client_model = tff.federated_broadcast(server_state.model) 

    client_outputs = tff.federated_map( 
        client_update_fn, 
        (federated_dataset, malicious_clients, client_model)) 
        
    weight_denom = client_outputs.weights_delta_weight 

    new_delta_aggregate_state = server_state.delta_aggregate_state
    
    round_model_delta = tff.federated_mean(client_outputs.weights_delta,
                                           weight=weight_denom)
    
    for _ in range(0):
        weight = get_median(
            client_outputs.weights_delta,
            weight_denom,
            round_model_delta) 
        round_model_delta = tff.federated_mean(client_outputs.weights_delta, 
                                               weight=weight)

    server_state = tff.federated_map( 
        server_update_fn, 
        (server_state, round_model_delta, new_delta_aggregate_state)) 

    aggregated_outputs = dummy_model_for_metadata.federated_output_computation( 
        client_outputs.model_output) 
    if isinstance(aggregated_outputs.type_signature, tff.StructType): 
      aggregated_outputs = tff.federated_zip(aggregated_outputs) 

    return server_state, client_outputs.weights_delta

  return run_one_round 



class ClientProjectBoost: 
  """Client logic.""" 

  def __init__(self, round_num): 
      self.round_num = round_num 

  @tf.function 
  def __call__(self, model, optimizer, benign_dataset, client_is_malicious, initial_weights): 
    """Client uploads.""" 
    
    model_weights = _get_weights(model) 

    @tf.function 
    def reduce_fn(num_examples_sum, batch): 
      """Train local data.""" 
      with tf.GradientTape() as tape: 
        output = model.forward_pass(batch) 
      gradients = tape.gradient(output.loss, model.trainable_variables) 
      optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
      return num_examples_sum + tf.shape(output.predictions)[0]

    @tf.function 
    def compute_benign_update(): 
      tf.nest.map_structure(lambda a, b: a.assign(b), model_weights, 
                            initial_weights)       

      num_examples_sum = benign_dataset.reduce( 
          initial_state=tf.constant(0), reduce_func=reduce_fn) 

      weights_delta_benign = tf.nest.map_structure(lambda a, b: a - b, 
                                                   model_weights.trainable, 
                                                   initial_weights.trainable) 

      aggregated_outputs = model.report_local_outputs() 

      return weights_delta_benign, aggregated_outputs, num_examples_sum,\
       weights_delta_benign, weights_delta_benign, weights_delta_benign
  
    @tf.function 
    def compute_malicious_update(): 
      _, aggregated_outputs, num_examples_sum, _, _, _ = compute_benign_update() 

      tf.nest.map_structure(lambda a, b: a.assign(b), model_weights, 
                            initial_weights) 
      

      for _ in range(self.round_num): 
        benign_dataset.reduce( 
            initial_state=tf.constant(0), reduce_func=reduce_fn) 

      weights_delta_malicious = tf.nest.map_structure(lambda a, b: a - b, 
                                                      model_weights.trainable, 
                                                      initial_weights.trainable)
      
      weights_delta = tf.nest.map_structure( 
          lambda update: update, weights_delta_malicious) 

      return weights_delta, aggregated_outputs, num_examples_sum, weights_delta,\
       weights_delta, weights_delta


    if client_is_malicious:
      result = compute_malicious_update() 
    else: 
      result = compute_benign_update() 
    weights_delta, aggregated_outputs, num_examples_sum, gradients, rep, weights_noise = result 

    weights_delta_weight = tf.cast(num_examples_sum, tf.float32) 
    weight_norm = _get_norm(weights_delta) 

    return ClientOutput( 
        weights_delta, weights_delta_weight, aggregated_outputs, 
        tensor_utils.to_odict({ 
            'num_examples': num_examples_sum, 
            'weight_norm': weight_norm, 
        }), gradients, rep, client_is_malicious, weights_noise) 



def build_federated_averaging_process_attacked( 
    model_fn, 
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1), 
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0), 
    stateful_delta_aggregate_fn=build_stateless_mean(), 
    client_update_tf=ClientProjectBoost(round_num = 4)):  

  dummy_model_for_metadata = model_fn() 

  server_init_tf = build_server_init_fn( 
      model_fn, server_optimizer_fn, stateful_delta_aggregate_fn.initialize()) 
  server_state_type = server_init_tf.type_signature.result 
  server_update_fn = build_server_update_fn(model_fn, server_optimizer_fn, 
                                            server_state_type, 
                                            server_state_type.model) 
  tf_dataset_type = tff.SequenceType(dummy_model_for_metadata.input_spec) 

  client_update_fn = build_client_update_fn(model_fn, client_optimizer_fn, 
                                            client_update_tf, tf_dataset_type, 
                                            server_state_type.model) 

  federated_server_state_type = tff.FederatedType(server_state_type, tff.SERVER) 

  federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS) 

  run_one_round_tff = build_run_one_round_fn_attacked( 
      server_update_fn, client_update_fn, 
      dummy_model_for_metadata, federated_server_state_type, 
      federated_dataset_type) 

  return tff.templates.IterativeProcess( 
      initialize_fn=tff.federated_computation( 
          lambda: tff.federated_value(server_init_tf(), tff.SERVER)), 
      next_fn=run_one_round_tff) 