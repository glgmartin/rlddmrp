import numpy as np
import os
import pandas as pd
from scipy.stats import variation, linregress
from .sim_utils import make_triangular_func

def make_time_funcs(rng, set_up_bounds, op_bounds, num_machines, num_products):
    #variable operation and set-up time generation for products
    # create records of format (setup time func, operation time func) for each
    # machine-product couple, using triangular distribution functions
    time_funcs = []
    set_up_modes = rng.integers(set_up_bounds[0], set_up_bounds[1], num_machines)
    op_modes = rng.integers(op_bounds[0], op_bounds[1], num_machines)

    for i in range(num_machines):
        time_funcs.append([])
        set_up_mode = set_up_modes[i]
        op_mode = op_modes[i]
        for j in range(num_products):
            time_funcs[i].append(
                (
                    make_triangular_func(rng, set_up_mode-4, set_up_mode, set_up_mode+4),
                    make_triangular_func(rng, op_mode-1, op_mode, op_mode+1)))
    return time_funcs

def initialize_simulation(demands, final_products_number, days, averages, 
    past_percentage, rng):

        #initialize parameters

        initialization_dict = {}

        # initialization is here to represent the fact that the system has a past and
        # could be roughly designed to fit this past
        # we will use the generated demand as the past
        past_demands = [demand[:int(past_percentage * days)] \
            for demand in demands]
        
        simulation_demands = [demand[int(past_percentage * days)+1:] \
            for demand in demands]

        sim_length_in_days = len(simulation_demands[0])
        sim_length_mins = sim_length_in_days * 8 * 60

        # get the average values of the demands
        past_averages = [np.mean(demand) for demand in past_demands]
        # adding error term
        past_averages = [avg + rng.exponential(3) for avg in past_averages]
        # make sure they are still postive
        past_averages = [np.abs(avg) for avg in past_averages]

        # adding past averages for components and raw materials using the bill of materials
        # 1 finished product consumes 1 of each component and each component consumes
        # 1 raw material of its type

        # total daily sum of averages of finished products divided by 3 as chances are equal
        consumption_sum_of_avgs = sum(past_averages) / 3 

        # log the values as ADUs in the dictionnary
        components = ['r0', 'r1', 'r2']
        finished_products = ['p{}'.format(i) for i in range(final_products_number)]
        for component in components:
            initialization_dict[component] = {'adu': consumption_sum_of_avgs}
        for product, adu in zip(finished_products, past_averages):
            initialization_dict[product] = {'adu': adu}

        # defining moqs
        moq_range = [100 * i for i in range(1, 5)]
        moq_prob = 0.2
        component_moqs = []
        product_moqs = []
        for component in components:
            if rng.random() > (1 - moq_prob):
                component_moqs.append(rng.choice(moq_range))
            else:
                component_moqs.append(0)
        for product in finished_products:
            if rng.random() > (1 - moq_prob):
                product_moqs.append(rng.choice(moq_range))
            else:
                product_moqs.append(0)

        for component, moq in zip(components, component_moqs):
            initialization_dict[component]['moq'] = moq
        for product, moq in zip(finished_products, product_moqs):
            initialization_dict[product]['moq'] = moq

        # variability factor is decided according to past_demand variations
        # high variations imply higher variability factor
        # attribution is made linearly with regression
        # some variations can be null if not enough demand, set them to 0
        variations = [variation(past_demand) if sum(past_demand) != 0 else 0 for past_demand in past_demands]
        sorted_variations = sorted(variations) #to attribute variability factors roughly
        variability_factors_range = np.linspace(0.2, 0.8, final_products_number)
        # linear regression
        slope, intercept, *_ = linregress(sorted_variations, variability_factors_range)

        # appliying linear regression
        variability_factors = [elt * slope + intercept for elt in variations]

        # variability factors for raw materials and components are the average of the 
        # previous ones
        component_variability_factor = np.mean(variability_factors)

        # log the variability factors in the initialization dict
        for component in components:
            initialization_dict[component]['vf'] = component_variability_factor
        for product, vf in zip(finished_products, variability_factors):
            initialization_dict[product]['vf'] = vf

        # deciding Decoupled Lead Times
        # DLT is roughly decided to accomodate a production order of size 2 * ADU rounded up 
        # Operation time is between 5 and 30 timesteps (see below for details)
        lot_sizes = [np.ceil(((2 * avg) / 10)) * 10 for avg in past_averages]
        component_lot_sizes = np.ceil(((2 * consumption_sum_of_avgs) / 10)) * 10
        # get a rough idea of DLT by multiplying by the machine range but slightly higher
        decoupled_lead_times = [elt * np.random.randint(20, 40) for elt in lot_sizes]
        # adding error term
        decoupled_lead_times = [dlt + rng.poisson(5*8*60) for dlt in decoupled_lead_times]
        # production is 1 timestep per product for components
        component_dlt = component_lot_sizes * 1 #which is stupid to do but hey ...

        #deciding LTF
        # we have no real way to determine a value for LTFs properly
        # as a result we use a function of the past demand variation to estimate LTFs
        # the higher the demand variation, the lower the LTF starting from a base value of 0.
        scaling_factor = 0.8
        base_value = 0.25
        lead_time_factors = [base_value + scaling_factor * elt for elt in variability_factors]
        component_ltf = component_variability_factor * scaling_factor + base_value

        # log the LTF in the initialization dict
        for component in components:
            initialization_dict[component]['ltf'] = component_ltf
        for product, ltf in zip(finished_products, lead_time_factors):
            initialization_dict[product]['ltf'] = ltf
        
        #deciding initial levels
        # initial levels are here to make sure bufffers start with enough inventory to
        # last for some time without hitting zero too quickly
        # we size the initial levels to make sure buffers can hold for X * DLT days plus 2 days
        decoupled_lead_times_in_days = [np.ceil(dlt / (8 * 60)) + 2 for dlt in decoupled_lead_times]
        component_dlt_in_days = np.ceil(component_dlt / (8 * 60)) + 2
        initial_levels = [np.ceil(2.7 * adu * dlt) for adu, dlt in zip(past_averages, decoupled_lead_times_in_days)]
        component_initial_level = np.ceil(2.7 * component_dlt_in_days * consumption_sum_of_avgs)

        # log the dlt in the initialization dict (as we need it in days)
        for component in components:
            initialization_dict[component]['dlt'] = component_dlt_in_days
        for product, dlt in zip(finished_products, decoupled_lead_times_in_days):
            initialization_dict[product]['dlt'] = dlt

        # log the initial level in the initialization dict
        for component in components:
            initialization_dict[component]['level'] = component_initial_level
        for product, level in zip(finished_products, initial_levels):
            initialization_dict[product]['level'] = level

        # added 23/7/2020
        # spike management
        spike_pct = 1.0
        component_spike_horizon = np.ceil(spike_pct * component_dlt_in_days)
        product_spike_horizons = [np.ceil(spike_pct * dlt) for dlt in decoupled_lead_times_in_days]
        component_spike_threshold = np.ceil(
            0.5 * consumption_sum_of_avgs * component_dlt_in_days * component_ltf * (1 + component_variability_factor))
        product_spike_thresholds = [np.ceil(0.5 * past_avg * dlt * ltf * (1 + vf)) for past_avg, dlt, ltf, vf in zip(
            past_averages, decoupled_lead_times_in_days, lead_time_factors, variability_factors)]
        for component in components:
            initialization_dict[component]['spike_horizon'] = int(component_spike_horizon)
        for product, horizon in zip(finished_products, product_spike_horizons):
            initialization_dict[product]['spike_horizon'] = int(horizon)
        for component in components:
            initialization_dict[component]['spike_threshold'] = component_spike_threshold
        for product, threshold in zip(finished_products, product_spike_thresholds):
            initialization_dict[product]['spike_threshold'] = threshold

        return simulation_demands, sim_length_mins, initialization_dict

def initialize_test_simulation(demands, final_products_number, days, averages, 
    past_percentage, rng):

        #initialize parameters

        initialization_dict = {}

        # initialization is here to represent the fact that the system has a past and
        # could be roughly designed to fit this past
        # we will use the generated demand as the past
        past_demands = [demand[:int(past_percentage * days)] \
            for demand in demands]
        
        simulation_demands = [demand[int(past_percentage * days)+1:] \
            for demand in demands]

        sim_length_in_days = len(simulation_demands[0])
        sim_length_mins = sim_length_in_days * 8 * 60

        # get the average values of the demands
        past_averages = [np.mean(demand) for demand in past_demands]
        # adding error term
        past_averages = [avg + rng.exponential(3) for avg in past_averages]
        # make sure they are still postive
        past_averages = [np.abs(avg) for avg in past_averages]

        # adding past averages for components and raw materials using the bill of materials
        # 1 finished product consumes 1 of each component and each component consumes
        # 1 raw material of its type

        # total daily sum of averages of finished products divided by 3 as chances are equal
        consumption_sum_of_avgs = sum(past_averages) / 3 

        # only one component in the test configuration
        components = ['r0']
        finished_products = ['p{}'.format(i) for i in range(final_products_number)]
        for component in components:
            initialization_dict[component] = {'adu': consumption_sum_of_avgs}
        for product, adu in zip(finished_products, past_averages):
            initialization_dict[product] = {'adu': adu}

        # defining moqs
        moq_range = [100 * i for i in range(1, 5)]
        moq_prob = 0.2
        component_moqs = []
        product_moqs = []
        for component in components:
            if rng.random() > (1 - moq_prob):
                component_moqs.append(rng.choice(moq_range))
            else:
                component_moqs.append(0)
        for product in finished_products:
            if rng.random() > (1 - moq_prob):
                product_moqs.append(rng.choice(moq_range))
            else:
                product_moqs.append(0)

        for component, moq in zip(components, component_moqs):
            initialization_dict[component]['moq'] = moq
        for product, moq in zip(finished_products, product_moqs):
            initialization_dict[product]['moq'] = moq

        # variability factor is decided according to past_demand variations
        # high variations imply higher variability factor
        # attribution is made linearly with regression
        # some variations can be null if not enough demand, set them to 0
        variations = [variation(past_demand) if sum(past_demand) != 0 else 0 for past_demand in past_demands]
        sorted_variations = sorted(variations) #to attribute variability factors roughly
        variability_factors_range = np.linspace(0.2, 0.8, final_products_number)
        # linear regression
        # not needed in the test case
        # slope, intercept, *_ = linregress(sorted_variations, variability_factors_range)

        # applying linear regression
        # in test case, use directly the variation
        # variability_factors = [elt * slope + intercept for elt in variations]
        variability_factors = [elt + rng.random() for elt in variations]

        # variability factors for raw materials and components are the average of the 
        # previous ones
        component_variability_factor = np.mean(variability_factors)

        # log the variability factors in the initialization dict
        for component in components:
            initialization_dict[component]['vf'] = component_variability_factor
        for product, vf in zip(finished_products, variability_factors):
            initialization_dict[product]['vf'] = vf

        # deciding Decoupled Lead Times
        # DLT is roughly decided to accomodate a production order of size 2 * ADU rounded up 
        # Operation time is between 5 and 30 timesteps (see below for details)
        lot_sizes = [np.ceil(((2 * avg) / 10)) * 10 for avg in past_averages]
        component_lot_sizes = np.ceil(((2 * consumption_sum_of_avgs) / 10)) * 10
        # get a rough idea of DLT by multiplying by the machine range but slightly higher
        decoupled_lead_times = [elt * np.random.randint(20, 40) for elt in lot_sizes]
        # adding error term
        decoupled_lead_times = [dlt + rng.poisson(5*8*60) for dlt in decoupled_lead_times]
        # production is 1 timestep per product for components
        component_dlt = component_lot_sizes * 1 #which is stupid to do but hey ...

        #deciding LTF
        # we have no real way to determine a value for LTFs properly
        # as a result we use a function of the past demand variation to estimate LTFs
        # the higher the demand variation, the lower the LTF starting from a base value of 0.
        scaling_factor = 0.8
        base_value = 0.25
        lead_time_factors = [base_value + scaling_factor * elt for elt in variability_factors]
        component_ltf = component_variability_factor * scaling_factor + base_value

        # log the LTF in the initialization dict
        for component in components:
            initialization_dict[component]['ltf'] = component_ltf
        for product, ltf in zip(finished_products, lead_time_factors):
            initialization_dict[product]['ltf'] = ltf
        
        #deciding initial levels
        # initial levels are here to make sure bufffers start with enough inventory to
        # last for some time without hitting zero too quickly
        # we size the initial levels to make sure buffers can hold for X * DLT days plus 2 days
        decoupled_lead_times_in_days = [np.ceil(dlt / (8 * 60)) + 2 for dlt in decoupled_lead_times]
        component_dlt_in_days = np.ceil(component_dlt / (8 * 60)) + 2
        initial_levels = [np.ceil(2.7 * adu * dlt) for adu, dlt in zip(past_averages, decoupled_lead_times_in_days)]
        component_initial_level = np.ceil(2.7 * component_dlt_in_days * consumption_sum_of_avgs)

        # log the dlt in the initialization dict (as we need it in days)
        for component in components:
            initialization_dict[component]['dlt'] = component_dlt_in_days
        for product, dlt in zip(finished_products, decoupled_lead_times_in_days):
            initialization_dict[product]['dlt'] = dlt

        # log the initial level in the initialization dict
        for component in components:
            initialization_dict[component]['level'] = component_initial_level
        for product, level in zip(finished_products, initial_levels):
            initialization_dict[product]['level'] = level

        # added 23/7/2020
        # spike management
        spike_pct = 1.0
        component_spike_horizon = np.ceil(spike_pct * component_dlt_in_days)
        product_spike_horizons = [np.ceil(spike_pct * dlt) for dlt in decoupled_lead_times_in_days]
        component_spike_threshold = np.ceil(
            0.5 * consumption_sum_of_avgs * component_dlt_in_days * component_ltf * (1 + component_variability_factor))
        product_spike_thresholds = [np.ceil(0.5 * past_avg * dlt * ltf * (1 + vf)) for past_avg, dlt, ltf, vf in zip(
            past_averages, decoupled_lead_times_in_days, lead_time_factors, variability_factors)]
        for component in components:
            initialization_dict[component]['spike_horizon'] = int(component_spike_horizon)
        for product, horizon in zip(finished_products, product_spike_horizons):
            initialization_dict[product]['spike_horizon'] = int(horizon)
        for component in components:
            initialization_dict[component]['spike_threshold'] = component_spike_threshold
        for product, threshold in zip(finished_products, product_spike_thresholds):
            initialization_dict[product]['spike_threshold'] = threshold

        return simulation_demands, sim_length_mins, initialization_dict



def initialize_simulation_industrial(app_path, rng):
    param_path = app_path + '/rawdata/consolidation_parameters.csv'
    demand_path = app_path + '/rawdata/consolidation_demande.csv'
    stock_path = app_path + '/rawdata/consolidation_stock.csv'
    article_path = app_path + '/rawdata/consolisation_articles.csv'
    factor_path = app_path + '/rawdata/consolidation_facteurs.csv'
    routing_path = app_path + '/rawdata/consolidation_routing.csv'

    article_data = pd.read_csv(article_path)
    demand_data = pd.read_csv(demand_path)
    if not os.path.exists(factor_path):
        factor_data = None
    else:
        factor_data = pd.read_csv(factor_path)
    if os.path.exists(param_path):
        # use the user given parameters
        param_data = pd.read_csv(param_path, delimiter=';')

    stock_data = pd.read_csv(stock_path)
    routing_data = pd.read_csv(routing_path)

    past_horizon = param_data['past_horizon'][0]
    fc_horizon = param_data['forecast_horizon'][0]

    # find the number of products
    num_products = len(demand_data['product_id'].unique())
    # and their ids
    product_ids = list(demand_data['product_id'].unique())

    # compute initial parameters of the simulation
    sim_length_in_days = demand_data['date'].max() + 1
    sim_length_in_hours = sim_length_in_days * 24 #hyp des 3*8

    # hypothesis : dimensionning is done using parameters from the data
    # adus = demand_data.groupby('product_id')['demand'].mean()
    adus = article_data['AverageDailyUsage'].values
    for ix, adu in enumerate(adus):
        if adu == 0.0:
            adus[ix] = 1.0
    # if adu is 0 due to very low demands, set it to 1 to avoid dimensionning problems

    component = 'r0'
    # hyp : all products use the same component, the component has no moq
    initialization_dict = {}
    consumption_sum = sum(adus)
    initialization_dict[component] = {'adu': consumption_sum}
    initialization_dict[component]['moq'] = 0   

    dlts = article_data['decoupled_lead_time'].to_list()
    moqs = article_data['moq'].to_list()

    # initialize levels with stock data
    initial_levels = stock_data['OnHandInventory']

    spike_horizons = article_data['spike_horizon'].to_list()

    vfs = [factor_data[factor_data['product_id'] == product_id]['variability_factor'].values[0] for product_id in product_ids]
    ltfs = [factor_data[factor_data['product_id'] == product_id]['lead_time_factor'].values[0] for product_id in product_ids]

    component_vf = np.mean(vfs)
    initialization_dict[component]['vf'] = component_vf
    initialization_dict[component]['dlt'] = 30 #hypothèse
    initialization_dict[component]['ltf'] = 0.5 #hypothèse selon les dlts du cas réel
    initialization_dict[component]['level'] = np.ceil(3*consumption_sum*30)
    initialization_dict[component]['spike_horizon'] = 30 #hyp
    initialization_dict[component]['spike_threshold'] = 50 #hyp

    spike_thresholds = article_data['spike_threshold'].to_list()

    # only for finished products
    for i, product_id in enumerate(product_ids):
        initialization_dict['p{}'.format(product_id)] = {'adu': adus[i]}
        initialization_dict['p{}'.format(product_id)]['dlt'] = dlts[i]
        initialization_dict['p{}'.format(product_id)]['vf'] = vfs[i]
        initialization_dict['p{}'.format(product_id)]['ltf'] = ltfs[i]
        initialization_dict['p{}'.format(product_id)]['level'] = initial_levels[i]
        initialization_dict['p{}'.format(product_id)]['moq'] = moqs[i]
        initialization_dict['p{}'.format(product_id)]['spike_horizon'] = spike_horizons[i]
        initialization_dict['p{}'.format(product_id)]['spike_threshold'] = spike_thresholds[i]

    demand_array = pd.pivot(demand_data, index='product_id', columns='date', values='demand').values
    demand_array = [l.tolist() for l in demand_array]

    num_machines = len(routing_data['Resource'].unique())

    resources = np.array(routing_data['Resource'])
    capacities = np.array(routing_data['NumMachines'])
    machine_cap_dict = {}
    for resource, capacity in zip(resources, capacities):
        if resource not in machine_cap_dict.keys():
            machine_cap_dict[resource] = capacity
        else:
            if capacity != machine_cap_dict[resource]:
                print('error')

    return demand_array, sim_length_in_hours, initialization_dict, \
        past_horizon, fc_horizon, product_ids, num_machines, machine_cap_dict, \
        routing_data


