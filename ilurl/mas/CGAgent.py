import random
import numpy as np
from random import choice

class CGAgent:

    def __init__(self, name, possible_actions):
        self.name = name
        self.possible_actions = possible_actions
        self.payout_functions = []
        self.dependant_agents = []
