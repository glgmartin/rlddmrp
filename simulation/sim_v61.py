from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from queue import PriorityQueue
from abc import abstractmethod
import numpy as np 
# import .sim_utils
from . import sim_utils

class Event(object):
    def __init__(self, time: float) -> None:
        self.time: float = time

    @abstractmethod
    def execute(self, sim: Simulator) -> None:
        ...

    def __lt__(self, other: Event) -> bool:
        return self.time < other.time

    def __repr__(self) -> str:
        return 'Event ({})'.format(self.time)

class PriorityEvent(Event):
    def __init__(self, time: float, priority: Optional[int] = 9) -> None:
        Event.__init__(self, time)
        self.priority: Optional[int] = priority

    def __lt__(self, other: PriorityEvent) -> bool:
        if self.time == other.time:
            return self.priority < other.priority
        else:
            return self.time < other.time

    def __repr__(self) -> str:
        return 'Event ({})'.format(self.time)

class Event_Queue(object):
    def __init__(self) -> None:
        self.queue = PriorityQueue()

    def insert(self, element) -> None:
        self.queue.put(element)

    def remove_first(self):
        return self.queue.get()

    def remove(self, element):
        for i in range(self.size()):
            if self.queue.queue[i] == element:
                x = self.queue.queue[i]
                self.queue.queue.remove(x)
                return x
        return None

    def has_more(self) -> bool:
        return self.size() > 0

    def size(self) -> int:
        return self.queue.qsize()

class Simulator(object):
    def __init__(self, time: float = 0.0, end_time: Optional[float] = None) -> None:
        self.events: Event_Queue = Event_Queue()
        self.time: float = time
        self.end_time: Optional[float] = end_time

        #add end event
        if end_time is not None:
            self.insert(PriorityEvent(end_time, 0))

    def insert(self, e: Event) -> None:
        self.events.insert(e)

    def cancel(self, e: Event) -> None:
        self.events.remove(e)

    def now(self) -> float:
        return self.time

    def do_all_events(self) -> None:
        while self.events.has_more():
            e = self.events.remove_first()
            self.time = e.time
            # print(str((self.time / self.end_time) * 100) + ' %\r', end='')
            if self.end_time is not None and self.time >= self.end_time:
                break
            e.execute(self)

class MessageHandler(object):
    @abstractmethod
    def handle(self, msg: Message) -> None:
        print('Abstract method must be implemented!')

class Message(PriorityEvent, sim_utils.Prototype, sim_utils.Node):
    
    PUSH: int = 0
    PENDING: int = 1
    READY: int = 2
    FINISHED: int = 3

    counter: int = 0
    instances: List[Message] = []

    def __init__(
        self,
        time: float = 0.0,
        handler: Optional[MessageHandler] = None,
        command: Optional[Any] = None,
        source: Optional[Any] = None,
        qty: Optional[int] = None,
        prerequisites: List = [],
        status: int = PUSH,
        key: Optional[int] = None,
        product: Optional[Product] = None,
        duration: Optional[float] = None,
        setup: Optional[float] = None,
        simulator: Optional[Simulator] = None,
        priority: Optional[int] = 9,
        description: Optional[str] = None) -> None:

        PriorityEvent.__init__(self, time, priority)
        sim_utils.Node.__init__(self, key)
        sim_utils.Prototype()

        self.id: int = Message.counter

        self.handler : Optional[MessageHandler] = handler
        self.command: Optional[Any] = command
        self.source: Optional[Any] = source
        self.qty: Optional[int] = qty
        self.prerequisites: List = prerequisites
        self.status: int = status
        self.product: Optional[Product] = product
        self.duration: Optional[float] = duration
        self.setup: Optional[float] = setup
        self.wait_time: Optional[float] = None
        self.simulator: Optional[Simulator] = simulator
        self.description: Optional[str] = description

        Message.counter += 1
        Message.instances.append(self)

    @classmethod
    def extract(cls) -> List[Message]:
        return cls.instances

    def set_command(self, command, time: float) -> None:
        self.command = command
        self.time = time

    def execute(self, simulator: Simulator) -> None:    
        self.simulator = simulator
        if self.command is not None and self.handler is not None:
            self.handler.handle(self)

    def clone(self) -> Message:
        return Message(
            time= self.time,
            handler= self.handler,
            command= self.command,
            source= self.source,
            qty= self.qty,
            prerequisites= [],
            status= Message.PUSH,
            key= self.key,
            product= self.product
        )

    def is_ready(self) -> bool:
        if self.prerequisites == []:
            return True
        else:
            return all([message.status == Message.FINISHED for message in self.prerequisites])

    def __repr__(self) -> str:
        if self.description is not None:
            return self.description
        if self.command in ['serve_back_orders', 'update_adu', 'update_zones', 'decide_production',\
            'update_otd', 'send_demand']:
            return 'Mgt (time {}): {}, {}, status {}'.format(self.time, self.handler,\
                self.command, self.status)
        elif self.command in ['pull_from_stock', 'put_in_stock', 'make', 'pull_from_client']:
            return 'Act (time {}): {}, {}, status {} (qty {}, dur {})'.format(self.time,\
                self.handler, self.command, self.status, self.qty, self.duration)
        else:
            return self.command
        
class Pattern(object):
    def __init__(self, max_time: float) -> None:
        self.max_time: float = max_time

    def build(self) -> List:
        ...
        
class ConstantPattern(Pattern):
    def __init__(self, max_time: float, time_rate: float) -> None:
        Pattern.__init__(self, max_time)
        self.time_rate: float = time_rate

    def build(self):
        intervals: List = []
        time: float = 0.0
        next_time: float = 0.0
        while time <= self.max_time:
            intervals.append(next_time)
            next_time += self.time_rate
            time = next_time
        return intervals

class Generator(object):
    def __init__(
        self, 
        handler: Optional[MessageHandler] = None, 
        command: Optional[str] = None, 
        pattern: Optional[Pattern] = None,
        priority: Optional[int] = 9) -> None:
        self.handler: Optional[MessageHandler] = handler
        self.command: Optional[str] = command
        self.pattern: Optional[Pattern] = pattern
        self.priority: Optional[int] = priority     

    def insert(self, message: Message, simulator: Simulator) -> None:
        simulator.insert(message)

    def run(self, simulator: Simulator) -> None:
        intervals: List = self.pattern.build()
        msg_objects: List = []
        for time in intervals:
            msg_objects.append(
                Message(
                    time= time,
                    handler= self.handler,
                    command= self.command,
                    priority= self.priority
                )
            )
        for message in msg_objects:
            self.insert(message, simulator)

    def __repr__(self) -> str:
        return 'Message generator (handler: {}, command: {}, priority: {})'.format(
            self.handler, self.command, self.priority
        )

class ConstantGeneratorBuilder(object):
    def build(
        self, 
        max_time: float, 
        rate: float, 
        handler: Optional[MessageHandler], 
        command: Optional[str],
        priority: Optional[int]=9) -> Generator:
        constant_pattern: ConstantPattern = ConstantPattern(
            max_time= max_time,
            time_rate= rate
        )
        constant_generator: Generator = Generator(
            handler= handler,
            command= command,
            pattern= constant_pattern,
            priority= priority
        )
        return constant_generator

class Client(MessageHandler):

    instances: List[Client] = []

    @classmethod
    def extract(cls):
        return [instance.logger.extract() for instance in cls.instances]

    def __init__(self, mediator: Optional[Mediator] = None, target: Optional[Any] = None, 
    demands: List = [], spike_horizon=0, spike_threshold=0) -> None:
        MessageHandler.__init__(self)
        self.logger: sim_utils.Logger = sim_utils.Logger(['demands'])
        self.demands: List = demands
        self.mediator: Optional[Mediator] = mediator
        self.target: Optional[Any] = target
        self.spike_horizon = spike_horizon
        self.spike_threshold = spike_threshold

        Client.instances.append(self)

    def handle(self, msg: Message) -> None:
        if msg.command != 'send_demand':
            print('Command not recognized:', msg.command)
            return
        else:
            self.send_demand(msg)

    def qualify_demands(self):
        if self.spike_horizon > 0:
            valid_demands = self.demands[:self.spike_horizon]
        else:
            valid_demands = self.demands[:1]
        qualified_demands = [valid_demands.pop(0)]
        for i, demand in enumerate(valid_demands):
            if demand > self.spike_threshold:
                qualified_demands.append(demand)
                self.demands[i+1] = 0
        _ = self.demands.pop(0)
        return qualified_demands

    def send_demand(self, msg: Message) -> None:
        if len(self.demands) > 0:
            qualified_demands = self.qualify_demands()
            for demand_quantity in qualified_demands:
                if demand_quantity > 0:
                    demand = Message(
                        time= msg.simulator.now(),
                        handler= self.target,
                        command= 'pull_from_client',
                        source= self,
                        qty= demand_quantity,
                        prerequisites= [],
                        status= Message.READY
                    )
                    self.logger.log((demand.time, demand.qty), 'demands')
                    msg.simulator.insert(demand)
            return

    def __repr__(self) -> str:
        return 'Client for {}'.format(self.target)

class Stock(MessageHandler):

    instances: List[Stock] = []

    @classmethod
    def extract(cls):
        return [instance.logger.extract() for instance in cls.instances]

    def __init__(
        self,
        product: Product, 
        level: int, 
        mediator: Mediator,
        ltf: float,
        vf: float,
        adu: float,
        dlt: float,
        adu_update_method = None,
        otd_update_method = None,
        adu_window: int = 0,
        otd_window: int = 0,
        moq = 0) -> None:
        
        #inheritance management
        MessageHandler.__init__(self)
        #basic components for a stock
        self.logger: sim_utils.Logger = sim_utils.Logger(['fulfilled', 'demands',
             'wip', 'nfe', 'backorders', 'TORS', 'TORB', 'TOY', 'TOG', 'adu', 'level',
             'otd', 'orders', 'dlt', 'ltf'])
        self.mediator: Optional[Mediator] = mediator
        self.product: Product = product
        self.level: int = level
        self.nfe_level: int = level
        self.otd: float = 1.0
        self.demands_received: List = []
        self.backorders: List = []
        #specific ddmrp components
        self.ltf: float = ltf
        self.vf: float = vf
        self.adu: float = adu
        self.dlt: float = dlt
        self.demand_bucket: int = 0
        self.wip: int = 0
        self.nfe: int = level
        self.moq = moq
        # values must be initialized at creation
        self.red_safe_zone: int = 0
        self.red_base_zone: int = 0
        self.yellow_zone: int = 0
        self.green_zone: int = 0
        self.top_of_red_safe: int = 0
        self.top_of_red_base: int = 0
        self.top_of_yellow: int = 0
        self.top_of_green: int = 0
        self.update_zones(message= None)
        # functions: take history of (timestamp, value) as inputs and outputs a specific float
        self.adu_update_method = adu_update_method
        self.adu_window = adu_window
        self.otd_update_method = otd_update_method
        self.otd_window = otd_window

        Stock.instances.append(self)

    def update_otd(self, message: Message) -> None:
        # if function is not defined then do nothinh
        if self.otd_update_method is not None:
            temp_otd = self.otd_update_method(self.logger.logs['fulfilled'], self.otd_window)
            if temp_otd is not None:
                self.otd = temp_otd
        self.logger.log((message.time, self.otd), 'otd')

    def update_level(self, qty: Optional[int], message: Message) -> None:
        self.level += qty
        self.logger.log((message.time, self.level), 'level')

    def serve_demand(self, message: Message) -> None:
        temp_level: int  = self.level - message.qty
        # if enough inventory
        if temp_level >= 0:
            # update inventory
            self.update_level(-message.qty, message)
            # log the demand quantity of this timesteps bucket
            self.demand_bucket += message.qty
            self.logger.log((message.time, True), 'fulfilled')
            # if demand comes from another stock, warn it to raise the wip
            message.status = Message.FINISHED
            self.mediator.handle(message)
        else:
            # if not enough inventory
            self.backorders.append(message)
            self.demand_bucket += message.qty
            self.logger.log((message.time, False), 'fulfilled')
        self.nfe_level -= message.qty
        self.logger.log((message.time, message.qty), 'demands')
        self.update_otd(message)

    def serve_backorders(self, message: Message) -> None:
        total_backorders: int = 0
        for demand in self.backorders:
            total_backorders += demand.qty
        self.logger.log((message.time, total_backorders), 'backorders')
        while len(self.backorders) > 0:
            demand: Message = self.backorders[0]
            temp_level: int = self.level - demand.qty
            if temp_level > 0:
                self.update_level(-demand.qty, message)
                demand.status = Message.FINISHED
                self.mediator.handle(demand)
                _ = self.backorders.pop(0)                
            else:
                break

    def decide_production(self, message: Message) -> None:
        self.update_nfe(message)
        if self.nfe <= self.top_of_yellow:
            qty: int = self.top_of_green - self.nfe
            #log creation of the order 
            self.logger.log((message.time, qty), 'orders')
            self.update_wip(qty, message)
            self.send_demand(qty, message)

    def send_demand(self, qty: int, message: Message) -> None:
        demand: Message = Message(
            time= message.time,
            handler= self.mediator,
            prerequisites= [],
            status= Message.PUSH,
            qty= qty,
            source= self,
            command= 'pull_from_stock',
            product= self.product
        )
        self.mediator.handle(demand)

    def update_wip(self, qty: Optional[int], message: Message) -> None:
        self.wip += qty
        self.logger.log((message.time, self.wip), 'wip')

    def update_zones(self, message: Optional[Message]) -> None:
        self.red_safe_zone = np.ceil(self.adu * self.dlt * self.ltf * self.vf)
        self.red_base_zone = np.ceil(self.adu * self.dlt * self.ltf)
        self.yellow_zone = np.ceil(self.adu * self.dlt)
        self.green_zone = max(np.ceil(self.adu * self.dlt * self.ltf), self.moq)
        self.top_of_red_safe = self.red_safe_zone
        self.top_of_red_base = self.top_of_red_safe + self.red_base_zone
        self.top_of_yellow = self.top_of_red_base + self.yellow_zone
        self.top_of_green = self.top_of_yellow + self.green_zone
        #for alternative printing with tackplot
        if message is None:
            return
        else:
            time: float = message.simulator.now()
        self.logger.log((time, self.red_safe_zone), 'TORS')
        self.logger.log((time, self.red_base_zone), 'TORB')
        self.logger.log((time, self.yellow_zone), 'TOY')
        self.logger.log((time, self.green_zone), 'TOG')

    def update_nfe(self, message: Message) -> None:
        self.nfe = self.nfe_level - self.demand_bucket + self.wip
        self.demand_bucket = 0
        self.logger.log((message.time, self.nfe), 'nfe')

    def update_adu(self, message: Message) -> None:
        if self.adu_update_method is not None:
            temp_adu = self.adu_update_method(self.logger.logs['demands'], message.time)
            if temp_adu is not None:
                self.adu = temp_adu
        self.logger.log((message.time, self.adu), 'adu')

    def update_dlt(self, message: Message) -> None:
        temp_dlt, temp_ltf = self.mediator.update_ltf_dlt(message)
        if temp_dlt is not None:
            # modify this according to your timestep
            # self.dlt = temp_dlt / (8*60)# dlt in days
            self.dlt = temp_dlt / 24# dlt in days
        if temp_ltf is not None:
            self.ltf = temp_ltf
        self.logger.log((message.time, self.dlt), 'dlt')
        self.logger.log((message.time, self.ltf), 'ltf')

    def handle(self, message: Message) -> None:
        if message.command == 'serve_back_orders':
            self.serve_backorders(message)
            message.status = Message.FINISHED
        elif message.command == 'serve_demands':
            print('error')
        elif message.command == 'update_adu':
            self.update_adu(message)
            message.status = Message.FINISHED
        elif message.command == 'update_dlt':
            self.update_dlt(message)
            message.status = Message.FINISHED
        elif message.command == 'update_zones':
            self.update_zones(message)
            message.status = Message.FINISHED
        elif message.command == 'decide_production':
            self.decide_production(message)
            message.status = Message.FINISHED
        elif message.command == 'update_otd':
            self.update_otd(message)
            message.status = Message.FINISHED
        elif message.command == 'pull_from_stock' and message.status == Message.READY:
            #this is the first time you see an activity from another stock object put it in waiting list
            # at the planning level keep track of the received demands
            self.demands_received.append(message)
            # at the execution level serve the demand
            self.serve_demand(message)
        elif message.command == 'put_in_stock' and message.status == Message.READY:
            # stock receives an order ready to be added to level
            self.update_level(message.qty, message)
            self.update_wip(-message.qty, message)
            self.nfe_level += message.qty
            message.status = Message.FINISHED
            self.mediator.handle(message)
        elif message.command == 'pull_from_client':
            # this is the basic demand coming from client
           # at the planning level keep track of the received demands
            self.demands_received.append(message)
            # at the execution level serve the demand
            self.serve_demand(message)
        else:
            print('Unknown command', message.command)

    def __repr__(self) -> str:
        return 'Stock for {}'.format(self.product)

class Product(object):

    counter: int = 0

    def __init__(self, components: Dict[Product, int] = {}, routing: Optional[Routing] = None, 
        selling_price: Optional[float] = None, production_cost: Optional[float] = None,
        holding_pct: Optional[float] = None, acquisition_cost: Optional[float] = None,
        name: str = '') -> None:

        self.components: Dict[Product, int] = components
        self.routing: Optional[Routing] = routing
        self.id: int = Product.counter
        self.name: str = name
        self.selling_price: Optional[float] = selling_price
        self.production_cost: Optional[float] = production_cost
        self.holding_pct: Optional[float] = holding_pct
        self.acquisition_cost: Optional[float] = acquisition_cost

        Product.counter += 1

    def get_components(self) -> List:
        return [self.components.items()]

    def __repr__(self) -> str:
        return 'Product {} (id: {})'.format(self.name, self.id)

class Routing(object):
    def __init__(
        self, 
        template_messages: List[Message] = [], 
        builder: Optional[MessageSequenceBuilder] = None,
        activity_times: Optional[ActivityTimes] = None) -> None:
        self.template_messages: List[Message] = template_messages
        self.builder: Optional[MessageSequenceBuilder] = builder
        self.activity_times: Optional[ActivityTimes] = activity_times

    def get_copies(self, qty_modifier: Optional[int] = None) -> List[Message]:
        copies: List[Message] = self.builder.copy_sequence(self.template_messages)
        for message in copies:
            if qty_modifier is not None:
                #mainly for unit testing purposes
                message.qty *= qty_modifier
            if message.command == 'make':
                set_up, op_time = self.activity_times.get_times(message.handler, message.product)
                message.duration = set_up + op_time * message.qty
                message.setup = set_up
        return copies

    def __repr__(self) -> str:
        digraph = sim_utils.DirectedGraph(vertices=self.template_messages)
        for act in self.template_messages:
            if act.prerequisites is not None:
                preqs = act.prerequisites
                for preq in preqs:
                    digraph.add_edge_by_vertices(preq, act)
        return str(digraph)

class MessageSequenceBuilder(object):

    def copy_sequence(self, template_messages: List[Message]) -> List[Message]:
        # transform a template message sequence into related copy
        # last item of the sequence must be the last message in the chain

        # update the key of the original message so they can be linked correctly
        for i, message in enumerate(template_messages):
            message.key = i

        # store the visited nodes of the templates sequence
        visited: List[bool] = [False] * len(template_messages)

        # get starting point of the copy
        template_source: Message = template_messages[-1]
        cloned_source: Message = sim_utils.clone_graph(
            old_source= template_source,
            new_source= template_source.clone(),
            visited= visited
        )

        # get list of newly created clones
        explored: List[Message] = [cloned_source]
        frontier: List[Message] = [cloned_source]
        while frontier != []:
            current_node: Message = frontier.pop(0)
            successors: List[Message] = current_node.prerequisites
            for child in successors:
                if child in explored:
                    continue
                explored.append(child)
                frontier.append(child)
        return explored

class ActivityTimes(object):
    def __init__(self, times_dict: Dict[Machine, Dict[Product, Tuple[float, float]]] = {}) -> None:
        self.times_dict: Dict[Machine, Dict[Product, Tuple[float, float]]] = times_dict

    def write_times(self, machine: Machine, product: Product, 
        set_up_time: float, operation_time: float) -> None:
        if machine not in self.times_dict:
            self.times_dict[machine] = {}
        if product not in self.times_dict[machine]:
            self.times_dict[machine][product] = (set_up_time, operation_time)

    def get_times(self, machine: Machine, product: Product) -> Tuple[float, float]:
        return self.times_dict[machine][product]

    def get_set_up_time(self, machine: Machine, product: Product) -> float:
        return self.get_times(machine, product)[0]

    def get_operation_time(self, machine: Machine, product: Product) -> float:
        return self.get_times(machine, product)[1] 

class StochasticActivityTimes:
    def __init__(self, time_table={}, n=2):
        self.time_table = time_table
        self.n = n

    def register(self, product, resource, function):
        if product not in self.time_table.keys():
            self.time_table[product] = {}
        if resource not in self.time_table[product].keys():
            self.time_table[product][resource] = function

    def get_times(self, resource, product):
        float_times = (
            np.around(self.time_table[product][resource][0](), self.n),
            np.around(self.time_table[product][resource][1](), self.n))
        return float_times

class CustomQueue(object):
    def __init__(self, servers=None):
        self.queue = []
        self.servers = servers

    def insert(self, message):
        #when message arrives into the queue, initialize its waiting time
        message.wait_time = message.simulator.now()
        for server in self.servers:
            if server.is_available():
                # put waiting at zero
                message.wait_time = 0.0
                server.insert(message.simulator, message)
                return
        self.queue.append(message)

    def remove(self):
        # get msg 
        message = self.queue.pop(0)
        #update waiting time of the message
        message.wait_time = message.simulator.now() - message.wait_time
        return message

    def size(self):
        return len(self.queue)

class Server(PriorityEvent):
    def __init__(self, time=0.0, custom_queue=None, message_served=None, machine=None):
        PriorityEvent.__init__(self, time)
        self.message_served = message_served
        self.queue = custom_queue
        self.machine = machine

    def is_available(self):
        return self.message_served is None

    def insert(self, simulator, message):
        if not self.is_available():
            print('Busy working!')
        self.message_served = message
        self.time = message.simulator.now() + message.duration
        # self.machine.logger.log(message.duration, 'uptime')
        self.machine.logger.log((simulator.now(), message.wait_time), 'waittime')
        message.simulator.insert(self)

    def execute(self, simulator):
        self.machine.logger.log((simulator.now(), self.message_served.duration), 'uptime')
        self.message_served.status = Message.FINISHED
        self.machine.mediator.handle(self.message_served)
        self.message_served = None
        if self.queue.size() > 0:
            message = self.queue.remove()
            self.insert(simulator, message)

class Machine(MessageHandler):

    counter: int = 0
    instances: List[Machine] = []

    @classmethod
    def extract(cls):
        return [instance.logger.extract() for instance in cls.instances]

    def __init__(self, mediator=None, server_number=1):
        MessageHandler.__init__(self)
        self.id: int = Machine.counter
        self.logger: sim_utils.Logger = sim_utils.Logger(['uptime', 'waittime', 'arrivals'])
        self.queue: CustomQueue = CustomQueue()
        # self.server: Server = Server()
        self.servers = []
        for _ in range(server_number):
            server = Server()
            server.queue = self.queue
            server.machine = self
            self.servers.append(server)
        self.queue.servers = self.servers
        # self.server.queue = self.queue
        # self.server.machine = self
        self.mediator: Optional[Mediator] = mediator

        Machine.counter += 1
        Machine.instances.append(self)

    def handle(self, message: Message) -> None:
        self.insert(message)

    def insert(self, message: Message) -> None:
        #log arrival and quantity
        # self.logger.log((message.simulator.now(), message.qty), 'arrivals')
        self.queue.insert(message)

    def __repr__(self) -> str:
        return 'Machine {}'.format(self.id)

class Mediator(MessageHandler):
    def __init__(self, simulator: Optional[Simulator] = None, dlt_ltf_update_method = None) -> None:
        MessageHandler.__init__(self)
        self.index: int = 0
        self.messages_seen: Dict[int, Dict[Message, List[Message]]] = {}
        self.simulator: Optional[Simulator] = simulator
        self.dlt_ltf_update_method = dlt_ltf_update_method
        self.logger = sim_utils.Logger(['flow_time'])

    def handle(self, message: Message) -> None:
        if message.command == 'pull_from_stock' and message.status == Message.PUSH:
            # a pull from stock message has been created and comes from a stock object
            # it is an internal demand
            if message.product is None:
                print('Error: {} stock has no product attached to it!'.format(message.source))
                return
            if message.product.routing is None:
                # case where the product has no routing
                # either for testing or because it is a simplified case or a raw
                # material with no supplier
                # time of the new message is the same as the original message
                automatic_refill: Message = Message(
                    time= message.time + message.source.dlt,
                    handler= message.source,
                    command= 'put_in_stock',
                    source= self,
                    qty= message.qty,
                    status= Message.READY,
                    product= message.product,
                    simulator= self.simulator
                )
                self.simulator.insert(automatic_refill)
            else:
                # a demand message comes from an internal stock object and must be
                # analyzed to find bill of materials multiplication factors and
                # then sent to the final product routing object
                qty_modifier: int = message.qty
                # basic copies of template messages from the routing
                planned_messages: List[Message] = message.product.routing.get_copies(qty_modifier)
                for new_message in planned_messages:
                    # modify source of message to indicate original caller
                    new_message.source = message.source
                    # check which messages are doable now
                    if new_message.is_ready():
                        new_message.status = Message.READY
                        # time must be updated to now
                        new_message.time = self.simulator.now()
                        self.simulator.insert(new_message)
                    else:
                        # put others in pending
                        # time is untouched
                        new_message.status = Message.PENDING
                # update message duration to prepare for lead time measurements
                message.duration = 0.0
                #store the original demand and linked messages into mediator memory
                self.messages_seen[self.index] = {message: planned_messages}
                #update the message status
                message.status = Message.PENDING
                self.index += 1
        elif (message.command == 'pull_from_stock' or message.command == 'make' or message.command == 'put_in_stock') and message.status == Message.FINISHED:
            # mediator receives a finished pull from stock which has to be an internal message
            for _, sequence in self.messages_seen.items():
                for original_message, linked_messages in sequence.items():
                    if message in linked_messages:
                        #update the lead time
                        if message.duration is not None and message.wait_time is not None:
                            original_message.duration += message.duration
                        if message.wait_time is not None:
                            original_message.duration += message.wait_time
                        for linked_message in linked_messages:
                            if linked_message.status == Message.PENDING and linked_message.is_ready():
                                linked_message.status = Message.READY
                                #update time for the linked message
                                linked_message.time = self.simulator.now()
                                self.simulator.insert(linked_message)
                        # if all messages in the sequence are finished, then indicate the sequence is finished
                        sequence_status = [x.status == Message.FINISHED for x in linked_messages]
                        if all(sequence_status):
                            original_message.status = Message.FINISHED
                            # if the sequence is finished, record its lead time
                            self.logger.log((message.simulator.now(), original_message.product, original_message.duration), 'flow_time')
        elif message.command == 'pull_from_client' and message.status == Message.FINISHED:
            # nothing to do here
            return
        else:
            print('Work in progress', message.command, 'status:', message.status)

    def update_ltf_dlt(self, message: Message):
        if self.dlt_ltf_update_method is not None:
            temp_dlt, temp_ltf = self.dlt_ltf_update_method(self.logger.logs['flow_time'], message.handler.product)
            return temp_dlt, temp_ltf
        else:
            return None, None

    def extract_lead_times(self, source) -> List[float]:
        lead_times = []
        for _, sequence in self.messages_seen.items():
            for original_message in sequence.keys():
                if original_message.source == source and original_message.status == Message.FINISHED:
                    lead_times.append(original_message.duration)
        return lead_times

    def __repr__(self) -> str:
        return 'Mediator object'
