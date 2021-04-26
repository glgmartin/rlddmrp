#this is the main resource for the functions of the design of experiments

import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .sim_utils import average, transform_logs_discrete, \
    moving_average, forecast_moving_average

#factor functions
# P3: how to attribute profiles to products
# P0: how to control ADU 
# P1: how to control DLT and LTF
# P2: how to control VF

# P0 factor: ADU value definition
# level 1 : mean of past demands
# level 2 : moving average of 5 days
# level 3 : moving average of 20 days
# level 4 : moving average of X including forecasts

# P0 - level 1: ADU = mean of past demands
def adu_past_average(past_demands, *args):
    if past_demands == []:
        #the simulation framework keeps the old value only if the func returns 
        #something else than None
        return None
    else:
        series = transform_logs_discrete(past_demands)
        return average(series)

# P0 - level 2: ADU = moving average 5 days
def adu_moving_average5(past_demands, *args):
    # given past demands on logger format, return 5 days mv avg
    if past_demands == []:
        return None
    else:
        return moving_average(transform_logs_discrete(past_demands), 5, 1)

# P0 - level 3: ADU = moving average 20 days
def adu_moving_average20(past_demands, *args):
    if past_demands == []:
        return None
    else:
        return moving_average(transform_logs_discrete(past_demands), 20, 1)

# P0 - level 4: 
def adu_moving_average_with_forecasts(past_demands, *args):
    if past_demands == []:
        return None
    else:
        series = forecast_moving_average(transform_logs_discrete(past_demands), 20, 1, 20)
        return moving_average(series, 40, 1)

# P1: how to control DLT and LTF
# level 0: DLT constant (None) and LTF constant (None) as attributed in the initilization step
# level 1: DLT and LTF unique and fixed to meet the 95% probability of lead times
# level 2: DLT is average of lead times, LTF is linearly distributed
# level 3: DLT is exp.smoothed of lead times, LTF is linearly distributed
# level 4: DLT fixed to meet the 95% probability of time, LTF is linearly distributed

# P1 - level 1: DLT and LTF unique and fixed to meet the 95% probability of lead times
def dlt_unique_update(past_times):
    # given a distribution of all lead times, find the common DLT to meet the
    # 95th percentile of the distribution
    data = pd.DataFrame(past_times)
    # if we have less than one row of data, we cannot decide on percentile so we return None
    if len(data) <= 1:
        return None
    else:
        unique_dlt = np.percentile(data[2], 95)
        return unique_dlt # in days

def ltf_unique_update(past_times, min_ltf=0.1, max_ltf=0.9):
    # the longer the delay, the lower the factor
    # given a distribution of lead times find the unique LTF value
    # lead times distribution is given on the logs formaat as a list of tuples
    data = pd.DataFrame(past_times)
    # if we have less than one row of data, we cannot decide on percentile so we return None
    if len(data) <= 1:
        return None
    else:
        unique_dlt = np.percentile(data[2], 95)
        # we need to find a linear fit between the values of lead times and the lead time factors
        min_ltf = min_ltf
        max_ltf = max_ltf
        # unique_lead_times = sorted(data[1].unique())
        max_lead_time = data[2].max()
        min_lead_time = data[2].min()
        lt_range = np.linspace(min_lead_time, max_lead_time, 2)
        ltf_range = np.linspace(max_ltf, min_ltf, 2)
        slope, intercept, *_ = linregress(lt_range, ltf_range)
        # ltf_range = np.linspace(max_ltf, min_ltf, len(unique_lead_times))
        # slope, intercept, *_ = linregress(unique_lead_times, ltf_range)
        lead_time_factor = intercept + (slope * unique_dlt)
        return lead_time_factor

def dlt_ltf_unique_update(past_times, *args):
    # given time records, define dlt and ltf
    # time records represent the amount of time between order creation and order stocked
    dlt = dlt_unique_update(past_times)
    ltf = ltf_unique_update(past_times)
    return dlt, ltf

# P1 - level 2: DLT is average of lead times, LTF is linearly distributed
def average_dlt(past_times):
    #given lead times on dataframe format return the average
    return past_times.mean()

def select_average_dlt(past_times, product, *args):
    # from the message information, select only a subset of lead times
    # if we have no data, return None
    if past_times == []:
        return None
    else:
        data = pd.DataFrame(past_times)
        data = data[data[1] == product]
        # if filtered data is also empty return None
        if len(data) == 0:
            return None
        else:
            return average_dlt(data[2])

def linear_ltf(past_times, chosen_dlt, min_ltf=0.1, max_ltf=0.9):
    # given records of lead times and hard coded boundaries of lead time factors
    # give the lead time value with a linear regression
    # if we have no data, return None
    if past_times == []:
        return None
    # if the dlt chosen previously is none, then we have no real information for ltf
    elif chosen_dlt is None:
        return None
    else:
        data = pd.DataFrame(past_times)
        # if we have less than one row of data, we cannot decide on percentile so we return None
        if len(data) <= 1:
            return None
        else:
            # we need to find a linear fit between the values of lead times and the lead time factors
            min_ltf = min_ltf
            max_ltf = max_ltf
            max_lead_time = data[2].max()
            min_lead_time = data[2].min()
            lt_range = np.linspace(min_lead_time, max_lead_time, 2)
            ltf_range = np.linspace(max_ltf, min_ltf, 2)
            slope, intercept, *_ = linregress(lt_range, ltf_range)
            lead_time_factor = intercept + (slope * chosen_dlt)
            return lead_time_factor

def average_dlt_linear_ltf_update(past_times, product):
    dlt = select_average_dlt(past_times, product)
    ltf = linear_ltf(past_times, dlt)
    return dlt, ltf

# P1 - level 4: DLT fixed to meet the 95% probability of time, LTF is linearly distributed
def dlt_percentile_update(data):
    # given a distribution of all lead times, find the common DLT to meet the
    # 95th percentile of the distribution
    # if we have less than one row of data, we cannot decide on percentile so we return None
    if len(data) <= 1:
        return None
    else:
        return np.percentile(data, 95)

def select_percentile_dlt(past_times, product, *args):
    # from the message information, select only a subset of lead times
    # if we have no data, return None
    if past_times == []:
        return None
    else:
        data = pd.DataFrame(past_times)
        data = data[data[1] == product]
        # if filtered data is also empty return None
        if len(data) == 0:
            return None
        else:
            return dlt_percentile_update(data[2])

def dlt_percentile_ltf_linear_update(past_times, product):
    # given time records, define dlt and ltf
    # time records represent the amount of time between order creation and order stocked
    dlt = select_percentile_dlt(past_times, product)
    ltf = linear_ltf(past_times, dlt)
    return dlt, ltf

# P1 - level 5: DLT fixed to be unique at the 95th quantile and LTF linearly distributed
def dlt_unique_ltf_linear_update(past_times, product):
    dlt = dlt_unique_update(past_times)
    hidden_dlt = select_percentile_dlt(past_times, product)
    ltf = linear_ltf(past_times, hidden_dlt)
    return dlt, ltf

# P1 - level 8: DLT constant and LTF lineraly distributed
def dlt_constant_ltf_linear_update(past_times, product):
    hidden_dlt = select_percentile_dlt(past_times, product)
    ltf = linear_ltf(past_times, hidden_dlt)
    return None, ltf

# P1 - level 9: DLT unique percentile and LTF constant
def dlt_unique_ltf_constant(past_times, product):
    dlt = dlt_unique_update(past_times)
    return dlt, None

def dlt_percentile_ltf_constant(past_times, product):
    return select_percentile_dlt(past_times, product), None

def dlt_constant_ltf_unique(past_times, product):
    return None, ltf_unique_update(past_times)

def dlt_percentile_ltf_unique(past_times, product):
    return select_percentile_dlt(past_times, product), ltf_unique_update(past_times)
    