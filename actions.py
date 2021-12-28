import inspect

class Actions:


    #XXX use selenium
    def click(self):
        '''
        Click on the top link of google
        :return:
        '''


    def search(self, phrase):
        '''
        Search the given phrase in google. Phrase to search is decided by agent.
        For simplicity right now can just decide to search most common word on page.
        :param phrase: the word or phrase to search in google
        :return:
        '''


    def see(self):
        '''
        Get screenshot of browser
        :return:
        '''


    def read_html(self):
        '''
        Scrape the html of the current page and return it
        :return:
        '''


    def transfer_learn(self, other_architecture, ):
        '''
        Restructure architecture by getting other network's prediction of best child architecture
        and output / spawn a new agent

        :return:
        '''


    def sense_other_actions(self):
        '''
        Get the actions that the other agents have most recently taken
        :return:
        '''

def get_actions_num_actions(cls):
    actions = inspect.getmembers(cls, predicate=inspect.isfunction)
    return actions, len(actions)

# NOTE bad idea -- map to output the size of number of possible actions instead
# def map_actions_to_numbers(number, cls):
#     actions = inspect.getmembers(cls, predicate=inspect.isfunction)
#     num_actions = len(actions)
#     for action_name, method in actions:

    # This is a product of running Actor 2 and doesn't need to be specified as an action
    # def predict_next(self, state):
    #     '''
    #     Given the past state, predict what the current state should be
    #     :param state: the past state
    #     :return:
    #     '''
