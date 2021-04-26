import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .sim_v61 import Client, Stock, Product, Machine, Simulator, Mediator, MessageSequenceBuilder, Message, Routing, ConstantGeneratorBuilder, MessageHandler, StochasticActivityTimes
from .sim_utils import print_client_demands, print_stock_wip, print_stock_backorders, print_stock_nfe, print_stock_level, print_stock_adu, print_stock_zones, print_lead_times 
from .sim_utils import print_stock_ltf, transform_logs_discrete_with_dates

class GrouperAgent(MessageHandler):
    # grouper agent is a core object of grouped flow shop simulations
    def __init__(self, group_policy, ltf_policy) -> None:
        MessageHandler.__init__(self)
        self.group_policy = group_policy
        self.ltf_policy = ltf_policy
        self.boundaries = None

    def handle(self, msg: Message) -> None:
        if msg.command != 'update_groups':
            print('Command not recognized:', msg.command)
            return
        else:
            self.update_groups(msg)

    def update_groups(self, msg):
        groups, dlt_list, pseudo_dlt_list, self.boundaries = self.group_policy(
            Stock.instances, self.boundaries)
        if self.ltf_policy is not None:
            ltf_values = self.ltf_policy(groups, dlt_list, pseudo_dlt_list)
            # give products their values
            for ltf, stock in zip(ltf_values, Stock.instances):
                stock.ltf = ltf
        msg.status = Message.FINISHED

#helper environment class to gather the simulation pieces together
class GroupedFlowShopEnvironment(object):
    # a simulation environment is the collection of all simulation objects preprogrammed

    def __init__(self) -> None:
        self._seed = 0
        self.products = {}
        self.machines = {}
        self.stocks = {}
        self.clients = {}
        self.activity_times = None

    def reset(self):
        #make sure to start from clean state
        Client.instances = []
        Stock.instances = []
        Product.counter = 0
        Machine.counter = 0
        Machine.instances = []

    def seed(self, seed):
        self._seed = seed

    def make_sim_core(self, grouping_policy, ltf_policy, end_time=None):
        # create core elemnts of simulation
        self.simulator = Simulator(end_time=end_time)
        self.mediator = Mediator(self.simulator)
        self.seq_builder = MessageSequenceBuilder()
        self.grouper_agent = GrouperAgent(grouping_policy, ltf_policy)

    def make_products(self, final_products_number, bom_indices):
        # start by creating the raw materials
        r0 = Product(name='r0')
        # r1 = Product(name='r1')        
        # r2 = Product(name='r2')
        self.products['r0'] = r0
        # self.products['r1'] = r1
        # self.products['r2'] = r2
        #create the finished products
        # finished products are all the same and they go through all steps
        raw_materials = [r0]
        for i in range(final_products_number):
            self.products['p{}'.format(i)] = Product(
                name='p{}'.format(i),
                components={raw_materials[bom_indices[i]]: 1}
            )

    def make_machines(self):
        # we need machines
        self.machines['wc0'] = Machine(
                mediator= self.mediator
            )
        # self.machines['wc1'] = Machine(
        #         mediator= self.mediator
        #     )
        # self.machines['wc2'] = Machine(
        #         mediator= self.mediator
        #     )

    def make_times(self, time_funcs):
        # make the activity times tables, with the form self.machines[machine] : {self.finished_product: (setup, operation)}
        times_dict= {
            self.machines['wc0']: {},
            # self.machines['wc1']: {},
            # self.machines['wc2']: {},
        }

        for i, machine in enumerate(times_dict.values()):
            for j, product in enumerate([value for (key, value) in self.products.items() if key.startswith('p')]):
                machine[product] = (time_funcs[i][j][0], time_funcs[i][j][1])

        self.activity_times = StochasticActivityTimes()

        for machine_key, machine_list in times_dict.items():
            for product_key, product in machine_list.items():
                self.activity_times.register(product_key, machine_key, times_dict[machine_key][product_key])

    def make_stocks(self, initialization_dict):
        #stocks are semi-randomly initialized to make sure the starting position is roughly functional
        #raw material stocks are fixed
        stock_r0 = Stock(
            product= self.products['r0'],
            level= initialization_dict['r0']['level'],
            mediator= self.mediator,
            ltf= initialization_dict['r0']['ltf'],
            vf= initialization_dict['r0']['vf'],
            adu= initialization_dict['r0']['adu'],
            dlt= initialization_dict['r0']['dlt'], # DLT has to be defined in days, whatever the base timestep
            adu_window= 5 * 8 * 60, # 5 days in minutes
            otd_window= 5 * 8 * 60,
            moq = initialization_dict['r0']['moq']
        )
        # stock_r1 = Stock(
        #     product= self.products['r1'],
        #     level= initialization_dict['r1']['level'],
        #     mediator= self.mediator,
        #     ltf= initialization_dict['r1']['ltf'],
        #     vf= initialization_dict['r1']['vf'],
        #     adu= initialization_dict['r1']['adu'],
        #     dlt= initialization_dict['r1']['dlt'], # DLT has to be defined in days, whatever the base timestep
        #     adu_window= 5 * 8 * 60, # 5 days in minutes
        #     otd_window= 5 * 8 * 60,
        #     moq = initialization_dict['r1']['moq']
        # )
        # stock_r2 = Stock(
        #     product= self.products['r2'],
        #     level= initialization_dict['r2']['level'],
        #     mediator= self.mediator,
        #     ltf= initialization_dict['r2']['ltf'],
        #     vf= initialization_dict['r2']['vf'],
        #     adu= initialization_dict['r2']['adu'],
        #     dlt= initialization_dict['r2']['dlt'], # DLT has to be defined in days, whatever the base timestep
        #     adu_window= 5 * 8 * 60, # 5 days in minutes
        #     otd_window= 5 * 8 * 60,
        #     moq = initialization_dict['r2']['moq']
        # )
        self.stocks['sr0'] = stock_r0
        # self.stocks['sr1'] = stock_r1
        # self.stocks['sr2'] = stock_r2
        #for each finished produt make a random stock
        for i, _ in enumerate([product for product in self.products.keys() if product.startswith('p')]):
            self.stocks['sp{}'.format(i)] = Stock(
                product= self.products['p{}'.format(i)],
                level= initialization_dict['p{}'.format(i)]['level'],
                mediator= self.mediator,
                ltf= initialization_dict['p{}'.format(i)]['ltf'],
                vf= initialization_dict['p{}'.format(i)]['vf'],
                adu= initialization_dict['p{}'.format(i)]['adu'],
                dlt= initialization_dict['p{}'.format(i)]['dlt'], # DLT has to be defined in days, whatever the base timestep
                adu_window= 5 * 8 * 60, # 5 days in minutes
                otd_window= 5 * 8 * 60,
                moq = initialization_dict['p{}'.format(i)]['moq']
            )

    def make_routings(self, final_products_number):
        #routings for fixed parts first
        #raw materials have no routing
        # in a flow shop all the routings are the same 
        for i in range(final_products_number):
            #routing for pi
            # get the raw materials of the product
            raw_materials_names = [component.name for component in \
                self.products['p{}'.format(i)].components.keys()]
            pull_from_stock_msgs = []
            for name in raw_materials_names:
                pull_from_stock_msgs.append(
                    Message(
                    handler= self.stocks['s' + name],
                    command= 'pull_from_stock', #remove 1 r0 from its stock
                    qty= 1, #must precise the quantity, we do not copy from bill of materiel yet :(
                    description= 'pull {} from stock'.format(name)
                )
            )
            # make the production messages
            m1 = Message(
                handler= self.machines['wc0'],
                command= 'make',
                qty= 1, # make 1 p0 from all components
                product= self.products['p{}'.format(i)],
                prerequisites=pull_from_stock_msgs, #if all pull from stock are done, assemble on wc0
                description= 'make p{} from {}'.format(i, raw_materials_names)
            )
            # m2 = Message(
            #     handler= self.machines['wc1'],
            #     command= 'make',
            #     qty= 1, # make 1 p0 from all components
            #     product= self.products['p{}'.format(i)],
            #     prerequisites=[m1], #if all pull from stock are done, assemble on wc0
            #     description= 'make p{} from {}'.format(i, raw_materials_names)
            # )
            # m3 = Message(
            #     handler= self.machines['wc2'],
            #     command= 'make',
            #     qty= 1, # make 1 p0 from all components
            #     product= self.products['p{}'.format(i)],
            #     prerequisites=[m2], #if all pull from stock are done, assemble on wc0
            #     description= 'make p{} from {}'.format(i, raw_materials_names)
            # )
            m4 = Message(
                handler= self.stocks['sp{}'.format(i)],
                command= 'put_in_stock',
                qty= 1, # make 1 p0 from all components
                product= self.products['p{}'.format(i)],
                prerequisites=[m1], #if all pull from stock are done, assemble on wc0
                description= 'stock p{}'.format(i)
            )
            self.products['p{}'.format(i)].routing = Routing(
                template_messages= pull_from_stock_msgs + [m1, m4],
                builder= self.seq_builder,
                activity_times= self.activity_times
            ) 
    
    def make_clients(self, all_demands, initialization_dict):
        #make a client for each finished product, given the client history
        #1 finished product <> 1 history
        for product, dmds in zip([product for product in self.products.keys() \
            if product.startswith('p')], all_demands):
                if not isinstance(dmds, list):
                    dmds = dmds.tolist()
                self.clients['cli_{}'.format(product)] = Client(
                    mediator= self.mediator,
                    target= self.stocks['s{}'.format(product)],
                    demands= dmds, 
                    spike_horizon= initialization_dict[product]['spike_horizon'],
                    spike_threshold= initialization_dict[product]['spike_threshold']
                )
    
    def make_mgt_rules(self, max_time):
        # we create the rates to manage the events
        def constant_rate_event_runner(simulator, generator, max_time, rate, 
            handler, command, priority):
            generator.build(max_time, rate, handler, command, priority).run(simulator)
        # helper functions to plan management events
        constant_generator_builder = ConstantGeneratorBuilder()
        # rates in the system are daily
        daily_rate = 8 * 60 #a day in minutes
        weekly_rate = 5 * daily_rate
        # we create a concatenated list of handlers which will receive the management rules
        handlers = Stock.instances + Client.instances
        # we decide two lists of rules, depending on type of handler
        # events for stocks are listed in priority order

        stock_events = ['serve_back_orders', 'update_adu', 'update_dlt', 'update_zones', 'decide_production']
        # clients only have to release their demands
        client_events = ['send_demand']
        #event attribution
        for handler in handlers:
            if type(handler) == Stock:
                for priority, command in enumerate(stock_events):
                    if command in ['serve_back_orders', 'decide_production']:
                        constant_rate_event_runner(
                            self.simulator, constant_generator_builder, max_time, daily_rate,
                            handler, command, priority)
                    else:
                        constant_rate_event_runner(
                            self.simulator, constant_generator_builder, max_time, weekly_rate,
                            handler, command, priority)
            elif type(handler) == Client:
                for command in client_events:
                    constant_rate_event_runner(
                        self.simulator, constant_generator_builder,
                        max_time, daily_rate, handler, command, priority=0)
            else:
                print('error')
        # attribution grouping events to the grouper agent
        constant_rate_event_runner(self.simulator, constant_generator_builder, max_time,\
            weekly_rate, self.grouper_agent, 'update_groups', 2.5)

    def make_policies(self, adu_policy, dlt_policy):
        if adu_policy is not None:
            for _, stock in self.stocks.items():
                stock.adu_update_method = adu_policy
        if dlt_policy is not None:
            self.mediator.dlt_ltf_update_method = dlt_policy
    
    def make_simulation(self, bom_indices, all_demands, max_time, adu_policy, \
        dlt_policy, grouping_policy, ltf_policy, initialization_dict, time_funcs, \
        end_time=None, final_products_number=30):
        # precedence constraints of building a simulation
        # 0: Simulator / Mediator / SeqBuilder classes OK
        # 1: Products / Machines classes OK
        # 2: ActivityTimes / Stocks OK
        # 3: template messages for routing OK
        # 4: Routings / Clients OK
        # 5: the rest OK

        # step 0: make the core sim objects
        self.make_sim_core(grouping_policy, ltf_policy, end_time)

        # step 1: make the products and machines objects
        self.make_products(final_products_number, bom_indices)
        self.make_machines()

        # step 2: 
        self.make_times(time_funcs)
        self.make_stocks(initialization_dict)

        # step 3:
        self.make_routings(final_products_number)

        # step 4:
        self.make_clients(all_demands, initialization_dict)

        # step 5:
        self.make_mgt_rules(max_time)

        # step 6:
        self.make_policies(adu_policy, dlt_policy)

    def run(self):
        self.simulator.do_all_events()

    def render_episode(self, path=None, graph_output=False):
        fig = plt.figure(figsize=(10, 8))
        print_client_demands(Client, fig, path, graph_output)
        print_stock_wip(Stock, fig, path, graph_output)
        print_stock_backorders(Stock, fig, path, graph_output)
        print_stock_nfe(Stock, fig, path, graph_output)
        print_stock_level(Stock, fig, path, graph_output)
        print_stock_adu(Stock, fig, path, graph_output)
        print_stock_zones(Stock, fig, path, graph_output)
        print_lead_times(self.mediator, fig, path, graph_output)
        print_stock_ltf(Stock, fig, path, graph_output=False)

    def save_episode(self):
        # save a dataframe of everything to be printed 
        # saving client demand information
        client_df = pd.DataFrame()
        for client in Client.instances:
            inner_df = pd.DataFrame()
            logs = transform_logs_discrete_with_dates(client.logger.logs['demands'])
            inner_df['Time'] = logs[0]
            inner_df['Demand'] = logs[1]
            inner_df['Client'] = client 
            client_df = pd.concat([client_df, inner_df])
        client_df = client_df.reset_index(drop=True)
        # saving order sizes
        order_df = pd.DataFrame()
        for stock in Stock.instances:
            # breakpoint()
            inner_df = pd.DataFrame()
            logs = transform_logs_discrete_with_dates(stock.logger.logs['orders'])
            inner_df['Time'] = logs[0]
            inner_df['order'] = logs[1]
            inner_df['Product'] = stock.product 
            order_df = pd.concat([order_df, inner_df])
        order_df = order_df.reset_index(drop=True)
        # saving work in progress information
        wip_df = pd.DataFrame()
        for stock in Stock.instances:
            inner_df = pd.DataFrame()
            logs = transform_logs_discrete_with_dates(stock.logger.logs['wip'])
            inner_df['Time'] = logs[0]
            inner_df['WIP'] = logs[1]
            inner_df['Product'] = stock.product 
            wip_df = pd.concat([wip_df, inner_df])
        wip_df = wip_df.reset_index(drop=True)
        # saving backorders
        bo_df = pd.DataFrame()
        for stock in Stock.instances:
            inner_df = pd.DataFrame()
            logs = transform_logs_discrete_with_dates(stock.logger.logs['backorders'])
            inner_df['Time'] = logs[0]
            inner_df['Backorders'] = logs[1]
            inner_df['Product'] = stock.product 
            bo_df = pd.concat([bo_df, inner_df])
        bo_df = bo_df.reset_index(drop=True)
        # saving NFE
        nfe_df = pd.DataFrame()
        for stock in Stock.instances:
            inner_df = pd.DataFrame()
            logs = transform_logs_discrete_with_dates(stock.logger.logs['nfe'])
            inner_df['Time'] = logs[0]
            inner_df['NFE'] = logs[1]
            inner_df['Product'] = stock.product 
            nfe_df = pd.concat([nfe_df, inner_df])
        nfe_df = nfe_df.reset_index(drop=True)
        # saving ADU
        adu_df = pd.DataFrame()
        for stock in Stock.instances:
            inner_df = pd.DataFrame()
            logs = transform_logs_discrete_with_dates(stock.logger.logs['adu'])
            inner_df['Time'] = logs[0]
            inner_df['ADU'] = logs[1]
            inner_df['Product'] = stock.product 
            adu_df = pd.concat([adu_df, inner_df])
        adu_df = adu_df.reset_index(drop=True)
        # saving levels
        lvl_df = pd.DataFrame()
        for stock in Stock.instances:
            inner_df = pd.DataFrame()
            logs = transform_logs_discrete_with_dates(stock.logger.logs['level'])
            inner_df['Time'] = logs[0]
            inner_df['Inventory level'] = logs[1]
            inner_df['Product'] = stock.product 
            lvl_df = pd.concat([lvl_df, inner_df])
        lvl_df = lvl_df.reset_index(drop=True)
        # saving zones
        zones_df = pd.DataFrame()
        for stock in Stock.instances:
            inner_df = pd.DataFrame()
            logs_tors = transform_logs_discrete_with_dates(stock.logger.logs['TORS'])
            logs_torb = transform_logs_discrete_with_dates(stock.logger.logs['TORB'])
            logs_toy = transform_logs_discrete_with_dates(stock.logger.logs['TOY'])
            logs_tog = transform_logs_discrete_with_dates(stock.logger.logs['TOG'])
            inner_df['Time'] = logs_tors[0]
            inner_df['TORS'] = logs_tors[1]
            inner_df['TORB'] = logs_torb[1]
            inner_df['TOY'] = logs_toy[1]
            inner_df['TOG'] = logs_tog[1]
            inner_df['Product'] = stock.product 
            zones_df = pd.concat([zones_df, inner_df])
        zones_df = zones_df.reset_index(drop=True)
        # saving lead times
        lead_times_df = pd.DataFrame()
        logs = transform_logs_discrete_with_dates(self.mediator.logger.logs['lead_times'])
        lead_times_df['Lead times'] = logs[1]
        lead_times_df['Product'] = logs[0] 

        return client_df, wip_df, bo_df, nfe_df, adu_df, lvl_df, zones_df, lead_times_df, order_df
