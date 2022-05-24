"""Script for SPECTRE-based filters."""

import numpy as np
from sklearn.decomposition import PCA
from numpy import linalg as la
from scipy.linalg import expm
import math
import heapq
import robust_estimator as cov


mimages_per_client = 10


def detect_offline(updates, alpha):
    """Original SPECTRE based on the following paper:
    
    SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics.
        Jonathan Hayase, Weihao Kong, Raghav Somani, and Sewoong Oh.
        https://arxiv.org/abs/2104.11315
    """
    
    n = len(updates)
    
    # PCA
    pca = PCA(n_components=16)
    components = pca.fit_transform(updates)
    
    # centering
    mu = []
    for k in range(len(components[0])):
        mu.append(np.mean(components[:,k]))
    np.array(mu)
    for k in range(len(components)):
        components[k] = np.array(components[k]) - mu
        
    # cov estimate
    selected = cov.cov_estimation_iterate(components, alpha, iters=50)
    cov_h = np.cov(components[selected].T)
        
    # cov^{-1/2}
    v, Q = la.eig(cov_h)
    for i in range(len(v)):
        if v[i] == 0:
            v[i] = 1e-5
    V = np.diag(v**(-0.5))
    sigma = Q @ V @ la.inv(Q)
        
    # T2
    main_gradients = components @ sigma.T

    # QUE
    sigma_til = main_gradients[0].reshape(-1,1) @ [main_gradients[0]]
    for k in range(1, len(main_gradients)):
        sigma_til = sigma_til + main_gradients[k].reshape(-1,1) @ [main_gradients[k]]
    sigma_til = sigma_til / len(main_gradients)
        
    above = sigma_til
    for k in range(len(above)):
        above[k][k] = above[k][k] - 1
    above = 4 * above
        
    below = np.linalg.norm(sigma_til) - 1
        
    Q = expm(above/below)
        
    que = []
        
    for k in range(len(main_gradients)): 
        que.append([main_gradients[k]] @ Q @ main_gradients[k].reshape(-1,1) / np.trace(Q))
            
    que = np.array(que)
    
    # Threshold
    que_max = max(que)
    que_random = np.random.rand() * que_max
    
    h = math.ceil(1.5*alpha*n)
    que.sort()
    que_h_largest = que[-h]
    
    threshold = max(que_h_largest, que_random)
    
    return cov_h, threshold


def filters(components, cov_h, threshold):
    # cov^{-1/2}
    v, Q = la.eig(cov_h)
    for i in range(len(v)):
        if v[i] == 0:
            v[i] = 1e-5
    V = np.diag(v**(-0.5))
    sigma = Q @ V @ la.inv(Q)
    
    # T2
    main_gradients = components @ sigma.T

    # QUE
    sigma_til = main_gradients[0].reshape(-1,1) @ [main_gradients[0]]
    for k in range(1, len(main_gradients)):
        sigma_til = sigma_til + main_gradients[k].reshape(-1,1) @ [main_gradients[k]]
    sigma_til = sigma_til / len(main_gradients)
        
    above = sigma_til
    for k in range(len(above)):
        above[k][k] = above[k][k] - 1
    above = 4 * above
        
    below = np.linalg.norm(sigma_til) - 1
        
    Q = expm(above/below)
        
    que = []
        
    for k in range(len(main_gradients)):
        que.append([main_gradients[k]] @ Q @ main_gradients[k].reshape(-1,1) / np.trace(Q))
            
    que = np.array(que)
    
    # filter
    selected = []
    for i in range(len(que)):
        if que[i] < threshold:
            selected.append(i)
    components = components[selected]
            
    return components


def detect_online(updates, sigmas, thresholds, alpha):
  """Ouput robust estimation if the input sigmas and thresholds are empty arrays."""

  max_iter = 50
  n = len(updates)
  further_iter = True

  components = updates
    
  # centering
  mu = []
  for k in range(len(components[0])):
     mu.append(np.mean(components[:,k]))
  np.array(mu)
  for k in range(len(components)):
     components[k] = np.array(components[k]) - mu
        
  # filters
  for i in range(len(sigmas)):
      components = filters(components, sigmas[i], thresholds[i])

  cnt = 0
  while further_iter and cnt < max_iter:
    cnt += 1
    
    # cov estimate
    benign_num  = n - math.ceil(2*alpha*n)
    remain_malicious = len(components) - benign_num
    
    if remain_malicious <= 0:
        further_iter = False
        return sigmas, thresholds, np.mean(sigmas, axis = 0), sigmas[-1], further_iter, remain_malicious
    
    ratio = remain_malicious / len(components)
    selected = cov.cov_estimation_iterate(components, ratio, iters = 1)
    if len(selected) == len(components):
        further_iter = False
        if len(sigmas) == 0:
            sigmas.append(np.cov(components[int(alpha*n/1.5) :].T))
            thresholds.append(float('inf'))
        return sigmas, thresholds, np.mean(np.array(sigmas), axis = 0), sigmas[-1], further_iter, remain_malicious
    
    origl = len(components)
    cov_h = np.cov(components[selected].T)
    sigmas.append(cov_h)
        
    # cov^{-1/2}
    v, Q = la.eig(cov_h)
    for i in range(len(v)):
        if v[i] == 0:
            v[i] = 1e-5
    V = np.diag(v**(-0.5))
    sigma = Q @ V @ la.inv(Q)
        
    # T2
    main_gradients = components @ sigma.T

    # QUE
    sigma_til = main_gradients[0].reshape(-1,1) @ [main_gradients[0]]
    for k in range(1, len(main_gradients)):
        sigma_til = sigma_til + main_gradients[k].reshape(-1,1) @ [main_gradients[k]]
    sigma_til = sigma_til / len(main_gradients)
        
    above = sigma_til
    for k in range(len(above)):
        above[k][k] = above[k][k] - 1
    above = 4 * above
        
    below = np.linalg.norm(sigma_til) - 1
        
    Q = expm(above/below)
        
    que = []
        
    for k in range(len(main_gradients)):
        que.append([main_gradients[k]]@Q@main_gradients[k].reshape(-1,1) / np.trace(Q))
            
    que = np.array(que)
    
    # Threshold
    h = origl - len(selected)
    que.sort()
    que_h_largest = que[-h]
    
    threshold = que_h_largest
    thresholds.append(threshold)
    
    components = components[selected]
    
  return sigmas, thresholds, np.mean(np.array(sigmas), axis = 0), sigmas[-1], further_iter, remain_malicious


def get_threshold_online(updates, cov_h, alpha, ratio):
    """Get the QUE-score threshold."""

    n = len(updates)
    
    # cov^{-1/2}
    v, Q = la.eig(cov_h)
    for i in range(len(v)):
        if v[i] == 0:
            v[i] = 1e-5
    V = np.diag(v**(-0.5))
    sigma = Q @ V @ la.inv(Q)

    components = updates
    
    # centering
    mu = []
    for k in range(len(components[0])):
        mu.append(np.mean(components[:,k]))
    np.array(mu)
    for k in range(len(components)):
        components[k] = np.array(components[k]) - mu
        
    # T2
    main_gradients = components @ sigma.T

    # QUE
    sigma_til = main_gradients[0].reshape(-1,1) @ [main_gradients[0]]
    for k in range(1, len(main_gradients)):
        sigma_til = sigma_til + main_gradients[k].reshape(-1,1) @ [main_gradients[k]]
    sigma_til = sigma_til / len(main_gradients)
        
    above = sigma_til
    for k in range(len(above)):
        above[k][k] = above[k][k] - 1
    above = 4 * above
        
    below = np.linalg.norm(sigma_til) - 1
        
    Q = expm(above/below)
        
    que = []
        
    for k in range(len(main_gradients)):
        q = [main_gradients[k]]@Q@main_gradients[k].reshape(-1,1) / np.trace(Q)
        que.append(q[0][0])
            
    que = np.array(que)
    
    # Threshold
    h = math.ceil(ratio*alpha*n)
    que.sort()
    que_h_largest = que[-h]
    
    threshold = que_h_largest
    
    return threshold


def detect(updates, is_malicious, cov_h, threshold):
    """Detect malicious clients based on the given QUE threshold."""

    # cov^{-1/2}
    v, Q = la.eig(cov_h)
    for i in range(len(v)):
        if v[i] == 0:
            v[i] = 1e-5
    V = np.diag(v**(-0.5))
    sigma = Q @ V @ la.inv(Q)
    
    components = updates
    
    # centering
    mu = []
    for k in range(len(components[0])):
        mu.append(np.mean(components[:,k]))
    np.array(mu)
    for k in range(len(components)):
        components[k] = np.array(components[k]) - mu
        
    # T2
    main_gradients = components @ sigma.T

    # QUE
    sigma_til = main_gradients[0].reshape(-1,1) @ [main_gradients[0]]
    for k in range(1, len(main_gradients)):
        sigma_til = sigma_til + main_gradients[k].reshape(-1,1) @ [main_gradients[k]]
    sigma_til = sigma_til / len(main_gradients)
        
    above = sigma_til
    for k in range(len(above)):
        above[k][k] = above[k][k] - 1
    above = 4 * above
        
    below = np.linalg.norm(sigma_til) - 1
        
    Q = expm(above/below)
        
    que = []
        
    for k in range(len(main_gradients)):
        q = [main_gradients[k]]@Q@main_gradients[k].reshape(-1,1) / np.trace(Q)
        que.append(q[0][0])
            
    que = np.array(que)
    
    # detect accuracy
    detect_true = 0
    detect_false = 0
    
    user_benign = []
    user_malicious = []
    
    for i in range(len(que)):
        if que[i] >= threshold:
            if is_malicious[i]:
                detect_true += 1
                user_malicious.append(i)
            else:
                detect_false += 1
                user_benign.append(i)
    
    l_malicious = np.sum(np.array(is_malicious))

    return detect_true/l_malicious, detect_true, detect_false, user_malicious, user_benign


def detect_fix_num(updates, is_malicious, cov_h, alpha):
    """Regard 1.5n\alpha-fraction clients as adversarial users."""

    # cov^{-1/2}
    v, Q = la.eig(cov_h)
    for i in range(len(v)):
        if v[i] == 0:
            v[i] = 1e-5
    V = np.diag(v**(-0.5))
    sigma = Q @ V @ la.inv(Q)
    
    components = updates
    
    # centering
    mu = []
    for k in range(len(components[0])):
        mu.append(np.mean(components[:,k]))
    np.array(mu)
    for k in range(len(components)):
        components[k] = np.array(components[k]) - mu
        
    # T2
    main_gradients = components @ sigma.T

    # QUE
    sigma_til = main_gradients[0].reshape(-1,1) @ [main_gradients[0]]
    for k in range(1, len(main_gradients)):
        sigma_til = sigma_til + main_gradients[k].reshape(-1,1) @ [main_gradients[k]]
    sigma_til = sigma_til / len(main_gradients)
        
    above = sigma_til
    for k in range(len(above)):
        above[k][k] = above[k][k] - 1
    above = 4 * above
        
    below = np.linalg.norm(sigma_til) - 1
        
    Q = expm(above/below)
        
    que = []
        
    for k in range(len(main_gradients)):
        q = [main_gradients[k]]@Q@main_gradients[k].reshape(-1,1) / np.trace(Q)
        que.append(q[0][0])
            
    que = np.array(que)

    # get threshold
    que_t = []
    for i in range(len(que)):
        que_t.append(que[i])
    que_t = np.array(que_t)
    que_t.sort()
    
    n = len(components)
    malicious_num = int(1.5 * alpha * n)
    
    threshold = que_t[-malicious_num]
    
    # detect accuracy
    detect_true = 0
    detect_false = 0
    
    user_benign = []
    user_malicious = []
    
    for i in range(len(que)):
        if que[i] >= threshold:
            if is_malicious[i]:
                detect_true += 1
                user_malicious.append(i)
            else:
                detect_false += 1
                user_benign.append(i)
    
    l_malicious = np.sum(np.array(is_malicious))

    return detect_true/l_malicious, detect_true, detect_false, user_malicious, user_benign
