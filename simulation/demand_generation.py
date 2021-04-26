import numpy as np

def balanced_poisson_process(averages, lams, total_time, clip_threshold=0, rng=None):
    if not rng:
        rng = np.random.default_rng()
    demands = rng.poisson(lams, (total_time, len(lams)))
    if clip_threshold > 0:
        demands[demands > clip_threshold] = clip_threshold
    dp = np.ceil(np.divide(averages, lams))
    demands = np.multiply(demands, dp)
    return np.transpose(demands).tolist()

def sine(max_time, loc=0, amp=1, speed=1, phase=0):
    time = np.arange(0, max_time, 1)
    f = np.sin(phase + (time/speed))/2
    fmin = min(f)
    if fmin < 0:
        f = f - fmin
        fmin = min(f)
    f = amp * f
    fhat = np.mean(f)
    return f + np.abs(loc - fhat)

def nhpp(lam_funcs, total_time, rng):
    events = np.zeros((len(lam_funcs), total_time))
    for i, lam_func in enumerate(lam_funcs):
        events[i] = rng.poisson(lam_func, total_time)
    return np.transpose(events)

def non_homogenous_poisson_demands(averages, lam_funcs, lams, total_time, rng):
    demands = nhpp(lam_funcs, total_time, rng)
    dp = np.ceil(np.divide(averages, lams))
    demands = np.multiply(demands, dp)
    return np.transpose(demands).tolist()

def continuous_poisson_demands(averages, total_time, rng):
    demands = np.zeros((len(averages), total_time))
    for i, avg in enumerate(averages):
        demands[i] = rng.poisson(avg, total_time)
    return demands.tolist()

def constant_demands(averages, total_time):
    return np.full((len(averages), total_time), averages)