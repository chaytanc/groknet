from State import EnvState

class InternetEnv():

    def __init__(self):
        """
        Set up selenium + browser etc??
        """
        self.s = EnvState()

    def step(self, action, state_pred):
        """
        Performs an action in the environment and returns the next state, the reward from that action, and
        whether an episode is done or not (has the agent died?)
        #XXX :param action: click, search, prediction etc, list of Actions in order which they should be taken
        :param action: argmax of Actor1 output / most desired output
        :return: observation: next state
        :return: reward: value of taking the given action in this environment
        :return: done: indicates if time to reset environment
        """
        # perform given action to get next state
        action()
        #XXX working here to implement actions
        # take screenshot
        # screenshot_state =
        # scrape html
        # html_state =
        # get agents' health from envstate
        # get agents' recent actions from envstate
        # get number of other agents from envstate
        reward = self.calc_reward(state_pred, )


    def calc_reward(self, state_prediction, actual_state):
        """
        Cumulative difference between state_prediction and actual state
        :param state_prediction
        :param actual_state
        :return: reward
        """
        diff = abs(actual_state - state_prediction) * -1
        #XXX does just having max possible reward of 0 and every reward being negative work?
        # inversely correlate diff to reward on logarithmic scale -- extreme accuracy has diminishing returns
        return diff

    def render(self):
        """
        Visualizes environment
        :return:
        """


    def close(self):
        """
        Close visualization
        :return:
        """


    def reset(self):
        """
        Set environment back to default state (empty google search bar??)
        :return:
        """