from __future__ import annotations
from typing import List, Optional, TypeVar, Generic, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json
from collections import deque

# classes 
class Prototype(object):
    def clone(self) -> Prototype:
        ...

class Node(object):
    def __init__(self, key = None, adj = None): 
        self.key = key 
        self.adj = adj 

@dataclass
class Edge:
    u: int
    v: int

    def reversed(self) -> Edge:
        return Edge(self.v, self.u)

    def __str__(self) -> str:
        return f"{self.u} -> {self.v}"

V = TypeVar('V')

class DirectedGraph(Generic[V]):
    def __init__(self, vertices: List[V] = []) -> None:
        self.vertices: List[V] = vertices
        self.edges: List[List[V]] = [[] for vertex in vertices]
        self.parents: Dict[int, List[V]] = {self.vertices.index(vertex): [] for vertex in vertices}
    
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def edge_count(self) -> int:
        return sum(map(len, self.edges))

    def add_vertex(self, vertex) -> int:
        self.vertices.append(vertex)
        self.edges.append([])
        self.parents[self.index_of(vertex)]: List[V] = []
        return self.vertex_count - 1

    def add_edge(self, edge) -> None:
        self.edges[edge.u].append(edge)
        self.parents[edge.v].append(edge.u)

    def add_edge_by_indices(self, u, v) -> None:
        edge: Edge = Edge(u, v)
        self.add_edge(edge)

    def add_edge_by_vertices(self, first, second) -> None:
        u: int = self.vertices.index(first)
        v: int = self.vertices.index(second)
        self.add_edge_by_indices(u, v)

    def vertex_at(self, index) -> V:
        return self.vertices[index]

    def index_of(self, vertex) -> int:
        return self.vertices.index(vertex)

    def neighbors_for_index(self, index) -> List[V]:
        return list(map(self.vertex_at, [e.v for e in self.edges[index]]))

    def neighbors_for_vertex(self, vertex) -> List[V]:
        return self.neighbors_for_index(self.index_of(vertex))

    def edges_for_index(self, index) -> List[Edge]:
        return self.edges[index]

    def edges_for_vertex(self, vertex) -> List[Edge]:
        return self.edges_for_index(self.index_of(vertex))

    def parents_for_index(self, index) -> List[V]:
        return self.parents[index]

    def parents_for_vertex(self, vertex) -> List[V]:
        return self.parents[self.index_of(vertex)]

    def find_source(self) -> Optional[V]:
        for vertex in self.vertices:
            if self.parents_for_vertex(vertex) == []:
                return vertex
        return None

    def __str__(self) -> str:
        desc: str = ""
        for i in range(self.vertex_count):
            desc += f"{self.vertex_at(i)} -> {self.neighbors_for_index(i)}\n"
        return desc

class Logger(object):

    def __init__(self, events: List[str]) -> None:
        self.logs: Dict[str, List[float]] = {event: [] for event in events}

    def add_event_to_log(self, event: str) -> None:
        if event not in self.logs:
            self.logs[event]: List[float] = []

    def log(self, value: Any, event: str) -> None:
        self.logs[event].append(value)

    def extract(self) -> Dict[str, List[float]]:
        return self.logs

    def reset(self) -> None:
        self.logs: Dict[str, List[float]] = {key: [] for key in self.logs}

    def __str__(self) -> int:
        return str(self.logs.keys())

# functions
def transform_logs_continuous(logs, agg_func):
    # logs have a format of [(timsetamp, value), ...]
    # make the dataframe
    if logs == []:
        logs = [(0.0, 0.0)]
    df = pd.DataFrame(logs)
    #round the decimals to solve in between time steps problems
    df[0] = df[0].round()
    #group all steps with same time
    df = df.groupby([0]).agg(agg_func)
    #resample on integer frequency
    max_index = int(df.index.max())
    df = df.reindex(range(max_index+1), fill_value=0)
    return df[1]

def transform_logs_discrete(logs):
    if logs == []:
        return None
    # to transform logs that do not need resampling
    df = pd.DataFrame(logs)
    return df[1]

def transform_logs_discrete_with_dates(logs):
    if logs == []:
        return None
    # to transform logs that do not need resampling
    df = pd.DataFrame(logs)
    return df

def average(series):
    return series.mean()

def moving_average(series, window, min_periods):
    return series.rolling(window=window, min_periods=min_periods).mean().values[-1]

def forecast_average(series, steps):
    avg = average(series)
    start_index = series.index.max()+1
    yhat = pd.Series(avg, index=range(start_index, start_index + steps))
    return series.append(yhat)

def forecast_moving_average(series, window, min_periods, steps):
    yhat = series.copy()
    for i in range(steps):
        yhat.loc[i + len(series)] = moving_average(yhat, window, min_periods)
    return yhat

def clone_graph(old_source, new_source, visited):
    clone = None
    if visited[old_source.key] is False and old_source.prerequisites != []: 
        for old in old_source.prerequisites: 
            if clone is None or (clone is not None and clone.id != old.key): 
                clone = old.clone()
            new_source.prerequisites.append(clone) 
            clone_graph(old, clone, visited) 
            visited[old.key] = True
    return new_source

# utility for printing out sim results
def print_client_demands(client_cls, fig, path=None, graph_output=False):
    # fig = plt.gcf()
    # fig.set_size_inches(14, 10)
    for i, client in enumerate(client_cls.instances):
        logs = transform_logs_discrete_with_dates(client.logger.logs['demands'])
        plt.plot(logs[0], logs[1], label='Client for Product {}'.format(i))
    plt.legend(loc=1)
    plt.title('Client demands')
    plt.xlabel('Time (min)')
    plt.ylabel('Quantity')
    sns.despine(top=True, right=True)
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path + '/client_demands.png')
    if graph_output:
        plt.show()
    else:
        plt.clf()
        plt.close(fig)
        plt.close('all')

def print_stock_wip(stock_cls, fig, path=None, graph_output=False):
    # fig = plt.gcf()
    # fig.set_size_inches(14, 10)
    for stock in stock_cls.instances:
        logs = transform_logs_continuous(stock.logger.logs['wip'], sum)
        plt.plot(logs, label='WIP for {}'.format(stock))
    plt.legend(loc=1)
    plt.title('Work in process')
    plt.xlabel('Time (min)')
    sns.despine(top=True, right=True)
    plt.ylabel('Quantity')
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path + '/work_in_process.png')
    if graph_output:
        plt.show()    
    else:
        plt.clf()
        plt.close(fig)
        plt.close('all')

def print_stock_nfe(stock_cls, fig, path=None, graph_output=False):
    # fig = plt.gcf()
    # fig.set_size_inches(10, 8)
    for stock in stock_cls.instances:
        logs = transform_logs_discrete_with_dates(stock.logger.logs['nfe'])
        plt.plot(logs[0], logs[1], label=stock)
    plt.legend(loc=1)
    plt.title('Net Flow Equation')
    plt.xlabel('Time (min)')
    sns.despine(top=True, right=True)
    plt.ylabel('Quantity')
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path + '/nfe.png')
    if graph_output:
        plt.show()
    else:
        plt.clf()
        plt.close(fig)
        plt.close('all')

def print_stock_level(stock_cls, fig, path=None, graph_output=False):
    # fig = plt.gcf()
    # fig.set_size_inches(14, 10)
    for stock in stock_cls.instances:
        logs = transform_logs_discrete_with_dates(stock.logger.logs['level'])
        if logs is None:
            continue
        plt.plot(logs[0], logs[1], label=stock)
    plt.legend(loc=1)
    plt.title('Inventory level')
    plt.xlabel('Time (min)')
    sns.despine(top=True, right=True)
    plt.ylabel('Quantity')
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path + '/level.png')
    if graph_output:
        plt.show()
    else:
        plt.clf()
        plt.close(fig)
        plt.close('all')

def print_stock_adu(stock_cls, fig, path=None, graph_output=False):
    # fig = plt.gcf()
    # fig.set_size_inches(10, 8)
    for stock in stock_cls.instances:
        logs = transform_logs_discrete_with_dates(stock.logger.logs['adu'])
        plt.plot(logs[0], logs[1], label=stock)
    plt.legend(loc=1)
    plt.title('Average Daily Usage (ADU)')    
    plt.xlabel('Time (min)')
    sns.despine(top=True, right=True)
    plt.ylabel('Quantity')
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path + '/adu.png')
    if graph_output:
        plt.show()
    else:
        plt.clf()
        plt.close(fig)
        plt.close('all')

def print_stock_backorders(stock_cls, fig, path=None, graph_output=False):
    # fig = plt.gcf()
    # fig.set_size_inches(14, 10)
    for stock in stock_cls.instances:
        logs = transform_logs_discrete_with_dates(stock.logger.logs['backorders'])
        plt.plot(logs[0], logs[1], label=stock)
    plt.legend(loc=1)
    plt.title('Back orders over time')
    plt.xlabel('Time (min)')
    sns.despine(top=True, right=True)
    plt.ylabel('Quantity')
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path + '/backorders.png')
    if graph_output:
        plt.show()
    else:
        plt.clf()
        plt.close(fig)
        plt.close('all')

def print_stock_otd(stock_cls):
    for stock in stock_cls.instances:
        logs = transform_logs(stock.logger.logs['otd'])
        plt.plot(logs[0], logs[1], label=stock)
    plt.legend()
    plt.title('otd')
    # plt.savefig('C:/Users/Guillaume Martin/Cozy Drive/Devs/DD_sim_framework/otd.png')
    plt.show()

def print_stock_zones(stock_cls, fig, path=None, graph_output=False):
    for stock in stock_cls.instances:
    #     fig = plt.gcf()
        # fig.set_size_inches(14, 10)
        logs_tors = transform_logs_discrete_with_dates(stock.logger.logs['TORS'])
        logs_torb = transform_logs_discrete_with_dates(stock.logger.logs['TORB'])
        logs_toy = transform_logs_discrete_with_dates(stock.logger.logs['TOY'])
        logs_tog = transform_logs_discrete_with_dates(stock.logger.logs['TOG'])
        plt.stackplot(logs_tors[0], logs_tors[1], logs_torb[1], logs_toy[1], logs_tog[1], colors=['brown', 'red', 'yellow', 'green'])
        plt.title('Zone sizes history for {}'.format(stock))
        plt.xlabel('Quantity')
        sns.despine(top=True, right=True)
        plt.ylabel('Time (mins)')
        if path is not None:
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(path + '/{}_zones.png'.format(str(stock)[:-8]))
        if graph_output:
            plt.show()
        else:
            plt.clf()
            plt.close(fig)
            plt.close('all')

def print_lead_times(mediator_instance, fig, path=None, graph_output=False):
    logs = transform_logs_discrete_with_dates(mediator_instance.logger.logs['lead_times'])
    products = logs[0].unique().tolist()
    for product in products:
        # fig = plt.gcf()
        # fig.set_size_inches(14, 10)
        plt.hist(logs[logs[0] == product][1], bins=50)
        plt.title('Lead times distribution for {}'.format(product))
        plt.xlabel('Lead times (mins)')
        sns.despine(top=True, right=True)
        plt.ylabel('Count')
        if path is not None:
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(path + '/{}_lead_times.png'.format(str(product)[:-8]))
        if graph_output:
            plt.show()
        else:
            plt.clf()
            plt.close(fig)
            plt.close('all')

def print_stock_ltf(stock_cls, fig, path=None, graph_output=False):
    # fig = plt.gcf()
    # fig.set_size_inches(10, 8)
    for stock in stock_cls.instances:
        logs = transform_logs_discrete_with_dates(stock.logger.logs['ltf'])
        plt.plot(logs[0], logs[1], label=stock)
    plt.legend(loc=1)
    plt.title('Lead Time Factor (LTF)')    
    plt.xlabel('Time (min)')
    sns.despine(top=True, right=True)
    plt.ylabel('Percentage')
    if path is not None:
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(path + '/ltf.png')
    if graph_output:
        plt.show()
    else:
        plt.clf()
        plt.close(fig)
        plt.close('all')

class TimeGenerator:
    def __init__(self, left, mode, right, func):
        self.left = left
        self.mode = mode
        self.right = right
        self.func = func

    def __call__(self):
        return self.func()

    def __repr__(self):
        return f'Triangular ({self.left}, {self.mode}, {self.right})'

def make_triangular_func(rng, left, mode, right):
    def trifn():
        return rng.triangular(left, mode, right)
    return TimeGenerator(left, mode, right, trifn)

def save_to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)

# def make_moving_average(n):
#     def mavgn(past_demands, *args):
#         if past_demands == []: 
#             return None
#         else:
#             return moving_average(transform_logs_discrete(past_demands), n, 1)
#     return mavgn

def make_mavg_with_fc(n, p, m):
    def mavg_fc(past_demands, *args):
        if past_demands == []:
            return None
        else:
            series = forecast_moving_average(transform_logs_discrete(past_demands), p, 1, n)
            return moving_average(series, n, 1)
    return mavg_fc

def reduce_logs_industrial(logs, date, window, timestep=24):
    #custom made to sum all demand events that have the same date
    date = date // timestep
    # create all dates between date and date - window
    # existing_dates = [a for a,_ in logs]
    #aggregate time data, get all dates in the logs between date and date-window
    existing_demands = {}
    for t, d in reversed(logs):
        t = t // timestep
        if t >= date - window:
            if t not in existing_demands.keys():
                existing_demands[t] = 0
            existing_demands[t] += d
    # creates the real dates
    dates = [i for i in range(max(0, int(date)-window), int(date)+1)]
    # reduced logs four output
    reduced_logs = []
    for t in reversed(dates):
        if t in existing_demands.keys():
            reduced_logs.insert(0, existing_demands[t])
        else:
            reduced_logs.insert(0, 0)
    return reduced_logs

def reduce_logs(logs, date, window, timestep=8*60):
    # reformat date (it should be coming in the format of the simulation timesteps)
    date = int(date / timestep)
    # create all dates between date and date - window
    existing_dates = [a for a,_ in logs]
    dates = [i * timestep for i in range(max(0, date-window), date+1)]
    # reduced logs four output
    reduced_logs = []
    for _, t in enumerate(reversed(dates)):
        flt = float(t)
        if flt in existing_dates:
            ix = existing_dates.index(flt)
            reduced_logs.insert(0, logs[ix][1])
        else:
            reduced_logs.insert(0, 0)
    return reduced_logs

def make_moving_average_industrial(n, timestep=24):
    def mavgn(past_demands, date):
        if past_demands == []: 
            return None
        else:
            return np.mean(reduce_logs_industrial(past_demands, date, n, timestep))
    return mavgn

def make_moving_average(n, timestep=8*60):
    def mavgn(past_demands, date):
        if past_demands == []: 
            return None
        else:
            return np.mean(reduce_logs(past_demands, date, n, timestep))
    return mavgn

def SMAfc(data, p, m):
    # use only the data needed
    data = data[-m:]
    output = []
    # load existing data in deque
    buffer = deque(data)
    state = np.mean(buffer)
    for i in range(p):
        output.append(state)
        if len(output) < p:
            last_item = buffer.popleft()
            state = state + (state-last_item)/m
            buffer.append(state)
    return output

def make_SMAfc(n, p, m):
    def mavgfc(past_demands, date):
        if past_demands == []:
            return None
        else:
            data = reduce_logs(past_demands, date, n)
            fc = SMAfc(data, p, m)
            return np.mean(data+fc)
    return mavgfc

def make_SMAfc_industrial(n, p, m, timestep=24):
    def mavgfc(past_demands, date):
        if past_demands == []:
            return None
        else:
            data = reduce_logs_industrial(past_demands, date, n, timestep)
            fc = SMAfc(data, p, m)
            return np.mean(data+fc)
    return mavgfc