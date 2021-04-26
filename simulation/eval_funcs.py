import numpy as np
from copy import deepcopy
from .sim_utils import average, transform_logs_discrete, transform_logs_discrete_with_dates

def orders_in_full_percent(history):
    # given a history of delivery events, either True of False, return the percentage met
    # return np.mean(history)
    if len(history.index) == 0:
        return 1.0
    return average(history)

# main evaluation function of a simulation
def evaluate_reliability(stock_cls, window):
    # given a stock class, extract devliery in full information, identified by
    # 'fulfilled' in the logger, and return orders in full for each stock
    # extract all the logs
    logs = stock_cls.extract()
    # keep only the 'fulfilled' sections
    logs = [item['fulfilled'] for item in logs]
    # transform logs into simple history
    histories = []
    for item in logs:
        if window is None:
            histories.append(transform_logs_discrete(item))
        else:
            df = transform_logs_discrete_with_dates(item)
            histories.append(df[df[0] >= window][1])
    reliabilities = {}
    for stock, hist in zip(stock_cls.instances, histories):
        if hist is None:
            reliabilities['Reliability ' + str(stock)] = 1.0
        else:
            reliabilities['Reliability ' + str(stock)] = orders_in_full_percent(hist)
    return reliabilities

def evaluate_flow_times(stock_cls, mediator, window):
    product_keys = [x.product for x in stock_cls.instances if x.product.name.startswith('p')]
    logs = mediator.logger.logs['flow_time']
    flow_times = {}
    if window is None:
        window = 0
    for ts, product, flow_time in logs:
        if ts >= window:
            if product not in flow_times.keys():
                flow_times[product] = []
            flow_times[product].append(flow_time)
    for product in product_keys:
        if product not in flow_times.keys():
            flow_times[product] = []
    mean_flow_times = {}
    for key, item in flow_times.items():
        if key not in mean_flow_times.keys():
            mean_flow_times[f'Mean flow time for {key}'] = None
        if item == []:
            mean_flow_times[f'Mean flow time for {key}'] = 0
        else:
            mean_flow_times[f'Mean flow time for {key}'] = np.mean(item)
    return mean_flow_times

def evaluate_agility(env, max_time, window):
    # agility = taux d'utilisation d'une machine et éventuellement temps d'attente
    # updated 3-7-2020 to abandon lambda / mu approach
    if window is None:
        logs = {k: deepcopy(v.logger.logs) for k, v in env.machines.items()}
        for k, v in logs.items():
            for i, j in v.items():
                if j == []:
                    continue
                logs[k][i] = transform_logs_discrete(j).to_list()
    else:
        logs = {k: deepcopy(v.logger.logs) for k, v in env.machines.items()}
        for k, v in logs.items():
            for i, j in v.items():
                if j == []:
                    continue
                df = deepcopy(transform_logs_discrete_with_dates(j))
                logs[k][i] = df[df[0] >= window][1].to_list()

    caps = {k: len(v.servers) for k, v in env.machines.items()}
    usage = {}
    queue_length = {}

    for machine, log in logs.items():
        if log['uptime'] == []:
            usage['Usage ' + str(machine)] = 0
        else:
            if window is not None:
                open_time = max_time - window
            else:
                open_time = max_time
            usage['Usage ' + str(machine)] = np.sum(log['uptime']) / (caps[machine] * open_time)
        if log['waittime'] == []:
            queue_length['Average queue wait ' + str(machine)] = 0.0
        else:
            queue_length['Average queue wait ' + str(machine)] = np.mean(log['waittime'])
    return usage, queue_length

# added on 2-7-2020
def evaluate_nfe(stock_cls, window):
    # get the average nfe value for each product
    logs = stock_cls.extract()
    # keep only the 'fulfilled' sections
    logs = [item['nfe'] for item in logs]
    # transform logs into simple history
    histories = []
    for item in logs:
        if window is None:
            histories.append(transform_logs_discrete(item))
        else:
            if item == []:
                histories.append(None)
            else:
                df = transform_logs_discrete_with_dates(item)
                histories.append(df[df[0] >= window][1])
    nfes = {}
    for stock, hist in zip(stock_cls.instances, histories):
        if hist is None:
            nfes['Average NFE ' + str(stock)] = stock.nfe
        elif len(hist.index) == 0:
            nfes['Average NFE ' + str(stock)] = stock.nfe
        else:
            nfes['Average NFE ' + str(stock)] = np.mean(hist)
    return nfes

# added on 2-7-2020
def evaluate_on_hand(stock_cls, window):
    # get the average on hand value (to be compared with the nfe and the work in progress)
    logs = stock_cls.extract()
    logs = [item['level'] for item in logs]
    histories = []
    for item in logs:
        if window is None:
            histories.append(transform_logs_discrete(item))
        else:
            if item == []:
                histories.append(None)
            else:
                df = transform_logs_discrete_with_dates(item)
                histories.append(df[df[0] >= window][1])
    onhands = {}
    for stock, hist in zip(stock_cls.instances, histories):
        if hist is None:
            onhands['Average on hand ' + str(stock)] = stock.level
        elif len(hist.index) == 0:
            onhands['Average on hand ' + str(stock)] = stock.level
        else:
            onhands['Average on hand ' + str(stock)] = np.mean(hist)
    return onhands

def evaluate_simulation(env, stock_cls, machine_cls, mediator_obj, max_time, window):
    # return the evaluation result of a simulation in a dictionnary form
    # return [evaluate_reliability(stock_cls), evaluate_margin(stock_cls), \
        # evaluate_delays(machine_cls), evaluate_agility(machine_cls)]
    ag = evaluate_agility(env, max_time, window)
    return [
        evaluate_reliability(stock_cls, window), 
        evaluate_flow_times(stock_cls, mediator_obj, window),
        evaluate_nfe(stock_cls, window), 
        ag[0],  
        ag[1], 
        evaluate_on_hand(stock_cls, window)]

