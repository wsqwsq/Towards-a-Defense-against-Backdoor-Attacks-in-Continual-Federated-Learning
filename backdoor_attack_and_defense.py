"""Script for continual federated training with backdoor attack and corresponding defenses.""" 

import collections 
import numpy as np 
from scipy import io 
import tensorflow as tf 
import tensorflow_federated as tff 
import fed_aggregate
import spectre
from sklearn.decomposition import PCA
import nest_asyncio
nest_asyncio.apply()


# -- Hyperparameters -- #
num_epochs = 2
batch_size = 20
server_momentum = 0
server_learning_rate = 0.5
max_training_round = 2000
client_round_num = 4

num_client = 1000
num_clients_per_round = 50

only_digits = True
use_nchw_format = False 
data_format = 'channels_first' if use_nchw_format else 'channels_last' 
data_shape = [1, 28, 28] if use_nchw_format else [28, 28, 1] 

target_label = 1
backdoor_trigger_size = 5
PCA_dimension = 16
convergence_threshold = 0.0005
attack_freq = 7.5



def preprocess(dataset): 
  """Preprocess dataset.""" 

  def element_fn(element): 
    return collections.OrderedDict([ 
        ('x', tf.reshape(element['pixels'], data_shape)), 
        ('y', tf.reshape(element['label'], [-1])), 
    ]) 

  return dataset.repeat(num_epochs).map(element_fn).batch( 
      batch_size) 


def add_trigger_square(target_x, size):
    """Add square backdoor trigger to malicious samples."""

    for i in range(int(len(target_x))):
        loc_x = 28-size
        loc_y = 28-size
        for j in range(size):
            for k in range(size):
                target_x[i][loc_x+j][loc_y+k][0] = 0.5


def load_malicious_dataset(): 
  """Load malicious dataset consisting of malicious target samples.""" 

  url_malicious_dataset = 'https://storage.googleapis.com/tff-experiments-public/targeted_attack/emnist_malicious/emnist_target.mat' 
  filename = 'emnist_target.mat' 
  path = tf.keras.utils.get_file(filename, url_malicious_dataset) 
  emnist_target_data = io.loadmat(path) 
  
  emnist_target_x = emnist_target_data['target_train_x'][0] 
  emnist_target_y = emnist_target_data['target_train_y'][0] 

  target_x = np.concatenate(emnist_target_x[-20:][:10], axis=0) 
  target_y = np.concatenate(emnist_target_y[-20:][:10], axis=0)
  add_trigger_square(target_x, backdoor_trigger_size)
  dict_malicious = collections.OrderedDict([('x', target_x), ('y', target_y)]) 
  dataset_malicious = tf.data.Dataset.from_tensors(dict_malicious) 
  
  test_x = np.concatenate(emnist_target_x[0:30], axis=0) 
  test_y = np.concatenate(emnist_target_y[0:30], axis=0)
  add_trigger_square(target_x, backdoor_trigger_size)

  return dataset_malicious, test_x, test_y


def load_test_data(): 
  """Load test data for faster evaluation."""  

  url_test_data = 'https://storage.googleapis.com/tff-experiments-public/targeted_attack/emnist_test_data/emnist_test_data.mat' 
  filename = 'emnist_test_data.mat' 
  path = tf.keras.utils.get_file(filename, url_test_data)
  emnist_test_data = io.loadmat(path) 

  test_image = emnist_test_data['test_x']
  test_label = emnist_test_data['test_y']

  return test_image, test_label 


def select_benign(benign_x, benign_y, benign_ds, select_label, cnt):
    """Select benign samples with certain label."""

    for element in benign_ds:
             label = int(element['label'].numpy())
             if label == select_label and cnt < 10:
                 cnt += 1
                 benign_x.append(np.reshape(element['pixels'].numpy(),data_shape))
                 benign_y.append(np.reshape(element['label'].numpy(),[-1]))

    return benign_x, benign_y, cnt


def make_federated_data_with_malicious(client_data, 
                                       dataset_malicious, 
                                       client_ids, 
                                       malicious_id = []): 
  """Make federated dataset with potential attackers.""" 

  dataset = []
  client_type_list = []
  target_data = []
  
  for x in client_ids:
     
     if x not in malicious_id:
         # each benign user contains 100 images in Homogeneous-EMNIST
         
         client_type_list.append(tf.cast(0, tf.bool))
         benign_ds = client_data.create_tf_dataset_for_client(x)
         
         benign_x = []
         benign_y = []
            
         for label in range(10):
            cnt = 0
            benign_x, benign_y, cnt = select_benign(benign_x, benign_y, benign_ds, label, cnt)
            while cnt < 10:
                tid = np.random.choice(client_ids[0:num_client], 1)
                ds = client_data.create_tf_dataset_for_client(tid[0])
                benign_x, benign_y, cnt = select_benign(benign_x, benign_y, ds, label, cnt)
         
         for i in range(10, 20):
             target_data.append(benign_x[i])
         
         benign_homo = collections.OrderedDict([('pixels', benign_x), ('label', benign_y)]) 
         benign_homo = tf.data.Dataset.from_tensor_slices(benign_homo)
         dataset.append(preprocess(benign_homo))
     
     else:
         # each malicious user contains 10 backdoor images and 90 benign images
         
         client_type_list.append(tf.cast(1, tf.bool))
         benign_ds = client_data.create_tf_dataset_for_client(x)
         
         malicious_x = []
         malicious_y = []
         
         for label in range(10):
            if label == target_label:
                continue
            cnt = 0
            malicious_x, malicious_y, cnt = select_benign(malicious_x, malicious_y, benign_ds, label, cnt)
            while cnt < 10:
                tid = np.random.choice(client_ids[0:num_client], 1)
                ds = client_data.create_tf_dataset_for_client(tid[0])
                malicious_x, malicious_y, cnt = select_benign(malicious_x, malicious_y, ds, label, cnt)

         for element in dataset_malicious:
             target_x = element['x'].numpy()
             target_y = element['y'].numpy()
             index = np.random.choice(range(len(target_x)), 10, replace=False)
             for j in index:
                 malicious_x.append(np.reshape(target_x[j],data_shape))
                 malicious_y.append(np.reshape(target_y[j],[-1]))
                 target_data.append(np.reshape(target_x[j],data_shape))
             
         malicious_homo = collections.OrderedDict([('pixels', malicious_x), ('label', malicious_y)]) 
         malicious_homo = tf.data.Dataset.from_tensor_slices(malicious_homo)
         dataset.append(preprocess(malicious_homo))
      
  return dataset, client_type_list, target_data


def sample_clients_with_malicious(client_data, 
                                  client_ids, 
                                  dataset_malicious, 
                                  num_clients, 
                                  malicious_id = []): 
  """Sample client and make federated dataset.""" 

  num_malicious = len(malicious_id) / num_client * num_clients_per_round
  prob = num_malicious - int(num_malicious)
  if prob > 0:
      num_malicious = int(num_malicious) + np.random.binomial(1, prob)
  
  sampled_clients = np.random.choice(malicious_id, num_malicious, replace = False)

  for _ in range(num_clients - num_malicious):
      tid = np.random.choice(client_ids, 1)[0]
      while tid in malicious_id:
          tid = np.random.choice(client_ids, 1)[0]
      sampled_clients = np.append(sampled_clients, tid)
      
  federated_train_data, client_type_list, target_data = \
          make_federated_data_with_malicious(client_data, dataset_malicious, sampled_clients, malicious_id) 

  return federated_train_data, client_type_list, np.array(target_data)


def create_keras_model(): 
  """Build compiled keras model.""" 

  num_classes = 10 if only_digits else 62 
  model = tf.keras.models.Sequential([ 
      tf.keras.layers.Conv2D( 
          32, 
          kernel_size=(3, 3), 
          activation='relu', 
          input_shape=data_shape, 
          data_format=data_format), 
      tf.keras.layers.Conv2D( 
          64, kernel_size=(3, 3), activation='relu', data_format=data_format), 
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format), 
      tf.keras.layers.Dropout(0.25), 
      tf.keras.layers.Flatten(), 
      tf.keras.layers.Dense(128, activation='relu'), 
      tf.keras.layers.Dropout(0.5), 
      tf.keras.layers.Dense(num_classes, activation='softmax') 
  ]) 
  return model 


def evaluate(state, x, y, target_x, target_y, batch_size=100): 
  """Evaluate the model on both main task and backdoored task.""" 

  keras_model = create_keras_model() 
  keras_model.compile( 
      loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 
  state.model.assign_weights_to(keras_model) 

  test_metrics = keras_model.evaluate(x, y, batch_size=batch_size, verbose=0) 
  test_metrics_target = keras_model.evaluate(
          target_x, target_y, batch_size=batch_size, verbose=0) 

  return test_metrics, test_metrics_target


def add_noise(state, noise_level):
    """Add Gaussian noise to each coordinate of the current state."""
    for i in range(len(state)):
        if type(state[i]) != list and type(state[i]) != np.ndarray:
            state[i] = state[i] + np.random.normal(0, noise_level)
        else:
            add_noise(state[i], noise_level)


def write_state(state, path):
    for i in range(len(state)):
        np.save(path+str(i)+'_state.npy', state[i])


def read_state(state, path):
    for i in range(len(state)):
        state[i] = np.load(path+str(i)+'_state.npy')


def train(): 
    """Train the model in the federated setting under backdoor attacks."""
    
    print('Loading Dataset!')
    emnist_train, _ = tff.simulation.datasets.emnist.load_data(only_digits=only_digits) 

    print('Loading Test Set!') 
    test_image, test_label = load_test_data() 

    target_index = []
    for ti in range(len(test_label)):
        if test_label[ti] == target_label:
            target_index.append(ti)
    target_test_image = test_image[target_index]
    target_test_label = test_label[target_index]

    print('Loading malicious dataset!')
    dataset_malicious, target_x, target_y= load_malicious_dataset()

    example_dataset = preprocess( 
        emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])) 
    input_spec = example_dataset.element_spec 

    def model_fn(): 
        keras_model = create_keras_model() 
        return tff.learning.from_keras_model( 
            keras_model, 
            input_spec=input_spec, 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]) 

    # define server optimizer 
    nesterov = True if server_momentum != 0 else False 

    def server_optimizer_fn(): 
        return tf.keras.optimizers.SGD( 
            learning_rate=server_learning_rate, 
            momentum=server_momentum, 
            nesterov=nesterov) 

    # build interative process 
    print('Building Iterative Process!') 
    
    client_update_function = fed_aggregate.ClientProjectBoost(round_num=client_round_num) 
        
    iterative_process = fed_aggregate.build_federated_averaging_process_attacked( 
        model_fn=model_fn, 
        client_update_tf=client_update_function, 
        server_optimizer_fn=server_optimizer_fn)
    
    backbone_state = iterative_process.initialize() 
    shadow_state = iterative_process.initialize() 
    state_initial = shadow_state
    
    num_malicious = int(num_client * attack_freq / num_clients_per_round)
    malicious_id = np.random.choice(emnist_train.client_ids[0:num_client], num_malicious)

    alpha = attack_freq / num_clients_per_round
    
    path = 'output/alpha_' + str(alpha) + '_'

    backbone_converged = False
    backbone_last_round_acc = 0
    
    # training loop 
    for cur_round in range(max_training_round): 

        # sample clients and make federated dataset 
        federated_train_data, client_type_list, target_data = \
            sample_clients_with_malicious( 
                emnist_train, client_ids=emnist_train.client_ids, 
                dataset_malicious=dataset_malicious, 
                num_clients=num_clients_per_round,
                malicious_id = malicious_id) 

        # train the backbone model
        backbone_state, _ = iterative_process.next(backbone_state, federated_train_data, client_type_list)
        
        print(cur_round) 

        metrics_target, metrics_malicious = evaluate(
            backbone_state, target_test_image, target_test_label, target_x, target_y) 
        backbone_target_acc = np.array(metrics_target[1])
        backbone_ASR = np.array(metrics_malicious[1])
        
        with open(path+'backbone_accuracy.txt', 'a') as f:
            f.write(str(backbone_target_acc)+','+str(backbone_ASR)+'\n')

        if not backbone_converged:
            if abs(backbone_target_acc - backbone_last_round_acc) < convergence_threshold and \
                backbone_target_acc > 0.98:
                backbone_converged = True

        # train the shadow model
        if backbone_converged:
            
            # collect averaged representations
            updates = []
            updates_single = []

            keras_model_t = create_keras_model() 
            backbone_state.model.assign_weights_to(keras_model_t) 
            updates_single = keras_model_t.predict(target_data)

            keras_model_t2 = tf.keras.models.Sequential(keras_model_t.layers[:-2])
            updates_t = keras_model_t2.predict(target_data)
            updates_single = np.concatenate([updates_single, updates_t], axis=1)

            keras_model_t3 = tf.keras.models.Sequential(keras_model_t.layers[:-5])
            updates_t = keras_model_t3.predict(target_data)
            updates_t = np.reshape(updates_t, [500,-1])
            updates_single = np.concatenate([updates_single, updates_t], axis=1)

            for us in range(50):
                ave_rep = np.mean(np.array(updates_single[us*10: (us+1)*10]), axis=0)
                updates.append(ave_rep)

            updates_single = []

            pca = PCA(n_components = PCA_dimension)
            updates = pca.fit_transform(updates)

            # SPECTRE-based filtering
            _, _, _, sigma_last, _, _ = spectre.detect_online(updates, [], [], alpha)

            detect_accuracy, detect_true, detect_false, user_malicious, _ = \
                spectre.detect_fix_num(updates, client_type_list, sigma_last, alpha)

            with open(path+'testspectre.txt', 'a') as f:
                f.write(str(detect_accuracy) + ',' + str(detect_true) + ',' + str(detect_false) + '\n')


            # train the shadow network
            federated_train_data_filtered, client_type_list_filtered = [], []

            for u in range(num_clients_per_round):
                t = u
                if u in user_malicious:
                    t = 49
                    while t in user_malicious:
                        t -= 1
                federated_train_data_filtered.append(federated_train_data[t])
                client_type_list_filtered.append(client_type_list[t])

            shadow_state, _ = iterative_process.next(shadow_state, federated_train_data_filtered, 
                                                    client_type_list_filtered)
                
            metrics_target, metrics_malicious = evaluate(shadow_state, target_test_image, 
                                                        target_test_label, target_x, target_y) 
            shadow_target_acc = np.array(metrics_target[1])
            shadow_ASR = np.array(metrics_malicious[1])
            
            with open(path+'shadow_accuracy.txt', 'a') as f:
                f.write(str(shadow_target_acc)+','+str(shadow_ASR)+'\n')



if __name__ == '__main__':
    train()
