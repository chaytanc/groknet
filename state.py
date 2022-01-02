import threading

# on scale of 500 total system resources
AGENT_RESOURCE_OVERHEAD = 1


class EnvState:
    def __init__(self):
        # List of Agents with ids, attractiveness
        # {agent id: agent}
        self.agents = {}
        # Restructured to be OOP with agents managing their own states
        # {agentid : health out of 500}
        # self.agent_attractiveness = {}
        # Start at full health but agent hungers and overhead should diminish without good predictions
        self.system_resources = 500
        self.lock = threading.Lock()

    def update_resources(self, agent):
        """
        Updates the agents in the environment and recalculates system health based on updates
        :param agent: The agent that recently updated its state and needs to inform EnvState of changes.
        :return:
        """
        self.check_rep()
        self.lock.acquire()
        id = agent.id
        # Apoptosis
        if agent.enable == -1:
            self.agents.pop(id)
            return
        # Agent starved
        if agent.hunger == 0:
            self.agents.pop(id)
            return
        agent_past_contribution = self.calc_agent_contrib(self.agents[id])
        agent_contrib = self.calc_agent_contrib(agent)
        change = agent_contrib - agent_past_contribution
        self.system_resources = self.system_resources + change
        self.lock.release()
        self.check_rep()

    # Want total contrib to be 0-500 but if attractiveness is unbounded to what degree
    # should a single agent be able to affect total system resources?
    # If one agent makes perfect predicitons, transfer mechanism should spread knowledge
    # to all agents which would maximize system resources
    # Then each agent normalized max contribution is 1/num_agents * 500
    # except minus cost of each agent (converges at one all knowing agent)
    # NOTE: may want to introduce random deaths of agents to make species more robust
    def calc_agent_contrib(self, agent):
        # (negative what is lacking to be full hunger)
        h = (500 - agent.hunger) * -1 / 500
        # Attractiveness should be 0 if hunger is not satiated
        a = agent.attractiveness / 500
        e = agent.enable
        contrib = (h + a) * e
        normalized = contrib * 500 / len(self.agents) - AGENT_RESOURCE_OVERHEAD
        return normalized

    def check_rep(self):
        assert (self.system_resources in range(0, 500))
