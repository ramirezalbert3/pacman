from collections import Counter
import random


class Agent():
    def __init__(self, alpha, epsilon, gamma):
        self.epsilon = epsilon  # (exploration prob)
        self.alpha = alpha  # (learning rate)
        self.gamma = gamma  # (discount rate)
        self.V = Counter()
        self.Q = Counter()
        self.Policy = Counter()
        self.states = Counter()

    def update(self, state, action, reward, nextState):
        '''
        state = action => nextState and reward transition
        Q-Value, visits counter, Value and Policy update here
        '''
        self.states[state] += 1  # Increase visits counter
        prevQ = self.Q[state, action]
        # Reward for ending in next state + expected reward from then on
        increaseQ = reward + self.gamma*self.V[nextState]
        # For alpha=0 no learning, for =1 no remembering
        self.Q[state, action] += self.alpha * (increaseQ - prevQ)
        # Exploration function
        # Not really, we are accumulating and learning this in the
        # steps above so not really substracted for visits>1
        self.Q[state, action] += BASE_EXPLFNC_REW / self.states[state]
        if (self.states[state] > 1):
            self.Q[state, action] -= BASE_EXPLFNC_REW / (self.states[state]-1)

    def act(self, state):  # pragma: no cover
        '''
        Perform an action and call update
        '''
        pass
        
    def explore(self):
        '''
        Exploration function here
        '''
        pass

    def initState(self, state, actionList):  # pragma: no cover
        '''
        Init random Q values for a given state
        on first visit
        '''
        if(self.states[state] == 0):
            for action in actionList:
                self.Q[state, action] = random.uniform(MIN_INIT_RAND, MAX_INIT_RAND)
        return

    def getReward(self, state):  # pragma: no cover
        '''
        State is defined in a list:
        Points, Lives, DiceList, RemainingRolls, OtherPlayers
        '''
        pass

    def getAction(self, state):  # pragma: no cover
        '''
        Compute the action to take in the current state
        With probability self.epsilon we should take a random action
        and take the best policy action otherwise
        '''
        pass

    def getLegalActions(self):  # pragma: no cover
        '''
        Compute the action to take in the current state
        With probability self.epsilon we should take a random action
        and take the best policy action otherwise
        '''
        pass

    def getQValue(self, state, action):
        '''
        Return Q-value for a given pair "state & action"
        '''
        return self.Q[state, action]

    def getValue(self, state):
        '''
        Returns max_action Q(state,action) over legal actions
        '''
        return self.V[state]

    def getPolicy(self, state):
        '''
        Return the best action to take in a state
        Update Policy and V
        '''
        legalActions = self.getLegalActions()
        for action in legalActions:
            if self.Q[state, action] > self.V[state]:
                self.V[state] = self.Q[state, action]
                self.Policy[state] = action
        return self.Policy[state]
