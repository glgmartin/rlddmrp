#this is the main resource for the grouping functions of the design of experiments

import numpy as np
from scipy.stats import linregress
from .sim_utils import transform_logs_discrete
from . import factor_funcs as dff

# utility function for binning
def custom_bin(data, previous_bins=None, num_bins=3):
    # bins 1D data
    if type(data) != np.array:
        data = np.array(data)
    max_boundary = data.max()
    min_boundary = data.min()
    bin_size = (max_boundary - min_boundary) / num_bins
    if previous_bins is None:
        bins = [min_boundary + bin_size * i for i in range(0, num_bins-1)]
    else:
        bins = previous_bins
    binned_data = np.digitize(data, bins=bins[:num_bins-1], right=True)
    return binned_data, max_boundary, min_boundary, bin_size

#utility functions to go from Stock instances to usabel data
def extract_data_from_stocks(stock_instances):
    # convert data from list of stock objects to dlts and adus
    dlt_list = [stock.dlt for stock in stock_instances]
    demand_histories = []
    for stock in stock_instances:
        logs = transform_logs_discrete(stock.logger.logs['demands'])
        if logs is not None:
            demand_histories.append(
                transform_logs_discrete(stock.logger.logs['demands']).tolist())
        else:
            demand_histories.append([])
    # demand_histories = [transform_logs_discrete(stock.logger.logs['demands']).tolist() \
        # for stock in stock_instances]
    return dlt_list, demand_histories

# grouping functions
# what is grouping in ddmrp? identifying common products and giving them the same attributes
# a grouping function is how you create groups
# an update function is how you update the groups you have made

# level 0: no grouping, done on another set of experiments
# level 1: static grouping from Ptak
# level 2: dynamic grouping from Ptak (boundaries move with time)
# level 3: grouping on order size and frequencies

# level 1: static grouping from Ptak, linear attribution of LTF
def ptak_static_grouping(stock_instances, previous_boundaries=None):
    # group products according to their DLT and adu variations
    # boundaries are established at the start of the simulation
    # adu categories are technically useless because we do not update variaibility factor
    dlt_list, _ = extract_data_from_stocks(stock_instances)
    if previous_boundaries is None:
        # set boundaries
        dlt_categories, max_dlt, min_dlt, dlt_bin_size = custom_bin(dlt_list)
        new_boundaries = [min_dlt + i * dlt_bin_size for i in range(3)]
        pseudo_dlt_list = [
            min_dlt + 0.5 * dlt_bin_size, 
            min_dlt + 1.5 * dlt_bin_size,
            min_dlt + 2.5 * dlt_bin_size]
        return dlt_categories, dlt_list, pseudo_dlt_list, new_boundaries
    else:
        dlt_categories, max_dlt, min_dlt, dlt_bin_size = custom_bin(dlt_list, previous_boundaries)
        pseudo_dlt_list = [
            min_dlt + 0.5 * dlt_bin_size, 
            min_dlt + 1.5 * dlt_bin_size,
            min_dlt + 2.5 * dlt_bin_size]
        return dlt_categories, dlt_list, pseudo_dlt_list, previous_boundaries

# level 2: dynamic grouping from Ptak (boundaries move with time)
def ptak_dynamic_grouping(stock_instances, *args):
    groups, dlt_list, pseudo_dlt_list, boundaries = ptak_static_grouping(stock_instances, None)
    return groups, dlt_list, pseudo_dlt_list, boundaries

# level 0: constant LTF 
# represented by the None function

# level 1: linear LTF
def grouped_linear_ltf(categories, dlt_list, pseudo_dlt_list, min_ltf=0.1, max_ltf=0.9):
    # comment attribuer un ltf linéaire si tous les dlt sont les mêmes ?
    ltf_range = np.linspace(max_ltf, min_ltf, 2)
    dlt_max = max(dlt_list)
    dlt_min = min(dlt_list)
    if dlt_max == dlt_min:
        slope = 0
        intercept = (max_ltf - min_ltf) / 2
    else:
        lt_range = np.linspace(dlt_min, dlt_max, 2)
        slope, intercept, *_ = linregress(lt_range, ltf_range)
    return [intercept + (slope * pseudo_dlt_list[dlt_category]) \
        for dlt_category in categories]

# level 2: unique LTF
def grouped_unique_ltf(categories, dlt_list, pseudo_dlt_list, min_ltf=0.1, max_ltf=0.9):
    unique_dlt = np.percentile(dlt_list, 95)
    # all pseudo dlt for all categories are replaced by the percentile chosen dlt
    pseudo_dlt_list = [unique_dlt for _ in pseudo_dlt_list]
    dlt_values = grouped_linear_ltf(categories, dlt_list, pseudo_dlt_list, \
        min_ltf=0.1, max_ltf=0.9)
    return dlt_values

# level 3: linear LTF min 05
def grouped_linear_ltf_min05(categories, dlt_list, pseudo_dlt_list):
    return grouped_linear_ltf(categories, dlt_list, pseudo_dlt_list, min_ltf=0.5)

# level 4: linear LTF max 05
def grouped_linear_ltf_max05(categories, dlt_list, pseudo_dlt_list):
    return grouped_linear_ltf(categories, dlt_list, pseudo_dlt_list, max_ltf=0.5)

# modified copies of dlt control functions
#level 1: DLT unique and fixed to meet the 95% probability of lead times
def grouped_dlt_unique_update(past_times, *args):
    # given time records, define dlt and ltf
    # time records represent the amount of time between order creation and order stocked
    dlt = dff.dlt_unique_update(past_times)
    return dlt, None

# level 2: DLT average
def grouped_dlt_average(past_times, product):
    return dff.select_average_dlt(past_times, product), None

# level 3: DLT percentile
def grouped_dlt_percentile(past_times, product):
    return dff.select_percentile_dlt(past_times, product), None

# level 4: DLT median
def grouped_dlt_median(past_times, product):
    return dff.select_median_dlt(past_times, product), None