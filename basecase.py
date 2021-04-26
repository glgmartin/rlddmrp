from simulation.sim_init import make_time_funcs, initialize_test_simulation
from simulation.demand_generation import constant_demands
from simulation.experiments import run_experiments as exp

#verif timestep
# verif adu time window

if __name__ == "__main__":
    import time
    import numpy as np
    import logging
    from logging.handlers import RotatingFileHandler
    import os

    # for logging purposes when executed on server
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    log_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'activity.log'))
    file_handler = RotatingFileHandler(log_path, 'a', 1000000, 1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    main_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    main_logger.addHandler(stream_handler)

    # all the seeds for replications
    # seeds = [128, 362, 145, 550, 310, 119, 662, 560, 466, 251]
    seeds = [128, ]

    # overall parameters
    years = 2
    final_products_number = 1
    past_percentage = 0.2

    # draw a base average consumption vector
    # daily capacity
    capacity = 480
    # aim for 80% usage ratio
    C = .5 * capacity
    # average throughput for 3 minutes per product
    throughput = C / 3
    # average demand
    Dm = throughput / final_products_number
    # draw the real averages
    base_rng = np.random.default_rng(seed=42)
    # averages = base_rng.poisson(Dm, final_products_number)
    averages = base_rng.exponential(Dm, final_products_number)

    num_machines = 1

    time_funcs = make_time_funcs(base_rng, [5, 15], [2, 4], num_machines, final_products_number)

    bom_indices = [0]

    tic = time.time()

    for seed in seeds[0:3]:
        # rng for all the experiments of this seed
        rng = np.random.default_rng(seed)

        # generate the demand for the seed
        #years of 12 months of 20 days
        # no spikes so clip demand
        total_length_days = years * 12 * 20 

        # in the basic case, transform average to integers for daily consumption
        averages = averages.astype(int)

        complete_demands= constant_demands(
            averages, total_length_days
        )

        # get initial values for objects in the simulation
        simulation_demands, sim_length_mins, initialization_dict = \
            initialize_test_simulation(demands=complete_demands, 
                final_products_number=final_products_number, 
                days=total_length_days, averages=averages, 
                past_percentage=past_percentage, rng=rng)

        # run grouped flow shop experiments
        exp(
            bom_indices, time_funcs, simulation_demands, sim_length_mins, 
            initialization_dict, seed, main_logger)

    main_logger.info(f'Running campaign took {(time.time() - tic)/60} mins')
