from mesa import Agent, Model
from mesa.time import RandomActivation, SimultaneousActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from pdb import set_trace


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B

def compute_wealth(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    return sum(agent_wealths)



class BoltzmannWealthModel(Model):
    """A simple model of an economy where agents exchange currency at random.
    All the agents begin with one unit of currency, and each time step can give
    a unit of currency to another agent. Note how, over time, this produces a
    highly skewed distribution of wealth.
    """

    def create_agent(self, agent):
        self.schedule.add(agent)

        # Add the agent to a random grid cell
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        self.grid.place_agent(agent, (x, y))

    def __init__(self, N=100, width=10, height=10):
        self.num_agents = N
        self.grid = MultiGrid(height, width, True)
        self.schedule = SimultaneousActivation(self)
        self.model_wealth = 0
        self.datacollector = DataCollector(
            model_reporters={"Gini": compute_gini}
        )
        # Create agents
        for i in range(self.num_agents):
            if i%10:
                resource = Resource(i, self)
                self.create_agent(resource)
            engineer = Engineer(i, self)
            self.create_agent(engineer)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()


class Engineer(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth -= 100
            self.wealth += 10

    def step(self):
        self.move()
        # if self.wealth > 0:
        #     self.give_money()


class Resource(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = -1

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 10
            # self.wealth = 10

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()

# class SoftwareEngineer(Engineer):

# class InfrastructureEngineer(Engineer):

# class MachineLearningEngineer(Engineer):

# class Data(Resource):

# class Compute(Resource):

# class Algorithm(Resource)
