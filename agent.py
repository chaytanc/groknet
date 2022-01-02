import threading

# XXX hyperparameter
RECENT_ACTION_BUFFER_SIZE = 1


class Agent:
    # Give it a name and a reference to the global environment state
    # NOTE: lock on agent self management allows for multiple processes to affect the agent
    # state in parallel (the agent could try several actions in the env at once on diff processors for ex)
    def __init__(self, id, state):
        self.state = state
        self.id = id
        self.hunger = 500
        self.attractiveness = 0
        # 1 indicates enabled, 0 indicates off, -1 indicates dead
        self.enable = 1
        self.recent_actions = []
        self.lock = threading.Lock()
        self.check_rep()

    def update_attractiveness(self, value):
        self.check_rep()
        self.lock.acquire()
        self.attractiveness = value
        self.update_state()
        self.check_rep()
        self.lock.release()
        self.check_rep()

    def update_hunger(self, value):
        self.check_rep()
        self.lock.acquire()
        self.hunger = value
        self.update_state()
        self.check_rep()
        self.lock.release()
        self.check_rep()

    def update_enable(self, value):
        self.check_rep()
        self.lock.acquire()
        # Apoptosis occurs if value == -1 / env will delete reference
        self.enable = value
        self.update_state()
        self.lock.release()
        self.check_rep()

    def add_recent_action(self, action):
        self.check_rep()
        self.lock.acquire()
        # Should never be greater than
        if len(self.recent_actions) == RECENT_ACTION_BUFFER_SIZE:
            # pop off the oldest action to make room for the newest
            self.recent_actions.pop(0)
        self.recent_actions.append(action)
        self.update_state()
        self.lock.release()
        self.check_rep()

    def update_state(self):
        '''
        Call after updating agent state in order to recalc system resources
        :return:
        '''
        self.state.update(self)

    def check_rep(self):
        assert (len(self.recent_actions) <= RECENT_ACTION_BUFFER_SIZE)
        assert (self.attractiveness in range(0, 500))
        assert (self.hunger in range(0, 500))
        assert (self.enable == 0 or self.enable == 1)
