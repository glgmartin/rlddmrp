from copy import deepcopy
import time
import pandas as pd
import os
from itertools import product
from simulation.sim_v61 import *
from .grouping_funcs import ptak_static_grouping
from .eval_funcs import evaluate_simulation
from .environment import GroupedFlowShopEnvironment
from .sim_utils import make_moving_average

def run_experiments(bom_indices, time_funcs, simulation_demands, sim_length_mins, 
            initialization_dict, seed, logger):

    data_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))

    #run all combinations for M replications
    # these parameters were used for the first experiments before reinforcement learning
    # uncomment at your own risks
    hyper_parameters_dict = {
        'group_funcs':{
            'Ptak static': ptak_static_grouping, 
            # 'Ptak dynamic': ptak_dynamic_grouping,
        },
        'ltf_funcs': {
            'LTF constant': None,
            # 'LTF linear': grouped_linear_ltf,
            # 'LTF unique percentile': grouped_unique_ltf, 
        },
        'adu_funcs': {
            # 'ADU constant': None,
            'ADU moving average 60 days': make_moving_average(60),
            # 'ADU moving average 60 days plus 20 forecasts': make_SMAfc(60, 20, 40)
        },
        'dlt_funcs': {
            'DLT constant': None,
            # 'DLT unique percentile': grouped_dlt_unique_update,
            # 'DLT percentile': grouped_dlt_percentile,
        },
    }

    # generate combinations
    combinations = product(
        hyper_parameters_dict['group_funcs'].keys(),
        hyper_parameters_dict['ltf_funcs'].keys(),
        hyper_parameters_dict['adu_funcs'].keys(),
        hyper_parameters_dict['dlt_funcs'].keys())

    gen_start = time.time()
    
    full_evals = []
    # window_evals = []
    # window = sim_length_mins - 60 * 8 * 60 # three months in minutes

    for keys in combinations:
        #create the environment
        env = GroupedFlowShopEnvironment()

        env.reset()

        group_key = keys[0]
        group_val = hyper_parameters_dict['group_funcs'][group_key]
        ltf_key = keys[1]
        ltf_val = hyper_parameters_dict['ltf_funcs'][ltf_key]
        adu_key = keys[2]
        adu_val = hyper_parameters_dict['adu_funcs'][adu_key]
        dlt_key = keys[3]
        dlt_val = hyper_parameters_dict['dlt_funcs'][dlt_key]

        logger.info(f'Starting {seed} {group_key} {ltf_key} {adu_key} {dlt_key} ...')

        start = time.time()

        #build the corresponding simulation
        env.make_simulation(
            bom_indices, deepcopy(simulation_demands), \
            sim_length_mins, adu_val, dlt_val, group_val, ltf_val, \
            initialization_dict, time_funcs, sim_length_mins + 1, \
            len(simulation_demands))

        # run the simulation
        env.run()

        full_eval_list = evaluate_simulation(env, Stock, Machine, env.mediator, 
            sim_length_mins+1, None)
        full_eval_list.append(
            {
                'group_func': group_key,
                'ltf_func': ltf_key, 
                'adu_func': adu_key,
                'dlt_func': dlt_key
            })
        full_evals.append(full_eval_list)

        logger.info(f'{group_key} {ltf_key} {adu_key} {dlt_key} time elapsed {(time.time() - start):.2f} sec')

    logger.info(f'Running all simulations took {((time.time() - gen_start) / 60):.2f} mins')

   # saving results
    full_evaluations_dict = {}
    for elt in full_evals:
        for elt_dict in elt:
            for key, value in elt_dict.items():
                if key not in full_evaluations_dict.keys():
                    full_evaluations_dict[key] = []
                full_evaluations_dict[key].append(value)

    df = pd.DataFrame.from_dict(full_evaluations_dict)
    df.to_csv(data_path + '/full_sim_result_grouped_flow_shop_seed{}.csv'.format(
        seed), index=False)