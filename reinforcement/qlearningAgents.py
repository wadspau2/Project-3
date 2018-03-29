# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Qvalues = util.Counter() # Initialize a dictionary for Q-values, key will be (state,action)

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        Qvalue = self.Qvalues.get((state,action))
        if Qvalue == None:
            Qvalue = 0.0
        return Qvalue


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        PossibleActions = self.getLegalActions(state)
        if PossibleActions is None:
            return 0.0
        else:
            Qmax = float("-inf")
            for a in PossibleActions:
                Qvalue = self.getQValue(state,a)
                if Qvalue >= Qmax:
                    Qmax = Qvalue
            return Qmax


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        PossibleActions = self.getLegalActions(state)
        if PossibleActions:
            Vmax = float("-inf")
            bestAction = None #'Stop'
            for a in PossibleActions:
                Qv = self.getQValue(state,a)
                if Qv >= Vmax:
                    Vmax = Qv
                    bestAction = a
            if bestAction == None:
                print PossibleActions
            return bestAction
        return None
        '''
        PossibleActions = self.getLegalActions(state)
        if len(PossibleActions) == 0:
            return None
        init_action = PossibleActions[0]
        init_Q = self.getQValue(state,init_action)
        for a in PossibleActions:
            Qvalue = self.getQValue(state,a)
            if Qvalue >= init_Q:
                init_Q = Qvalue
                init_action = a
        if init_Q == 0.0:
            return random.choice(PossibleActions)
        else:
            return init_action
        '''

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if legalActions is not None:
            result = util.flipCoin(self.epsilon) # Returns true or false, if random number is less than epsilon
            if result == True:
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(nextState) # possible actions to calculate Vpi(s_prime)
        sample = reward # Result of sample equation if s is terminal then R will not be modified
        #print possibleActions
        if len(possibleActions) != 0:
            Q_array = [] # This will collect possible Q values for each action
            for a in possibleActions:
                Q_array.append(self.getQValue(nextState,a)) # Calculate Q values for each action, save in array to extract max
            sample = reward + self.discount * max(Q_array) # sample now includes reward from s'
        Vpi_s = self.getQValue(state,action)
        self.Qvalues[(state,action)] = Vpi_s + self.alpha * (sample - Vpi_s)
        '''
        legalActions = self.getLegalActions(nextState)
        if legalActions is not None:
            Qmax = float("-inf")
            for a in legalActions:
                Qvalue = self.getQValue(nextState,a)
                if Qvalue >= Qmax:
                    Qmax = Qvalue
            Vpi_prime = Qmax
            R = reward
            Sample = R + (self.discount * Vpi_prime)
            Vpi = self.getQValue(state,action)
            self.Qvalues[(state,action)] = Vpi + self.alpha * (Sample - Vpi)
        else:
            Vpi = self.getQValue(state, action)
            self.Qvalues[(state,action)] = Vpi + self.alpha * (reward - Vpi)
        '''


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feature_list = self.featExtractor.getFeatures(state, action)
        q_value = 0.0 # self.Qvalues.get((state, action))
        for feature in feature_list:
            q_value = q_value + self.weights[feature] * feature_list[feature] # dot product???
        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        r = reward
        a = self.alpha
        d = self.discount
        diff = (r + (d * self.getValue(nextState)) - self.getQValue(state,action))
        feature_list = self.featExtractor.getFeatures(state, action)
        for feature in feature_list.keys():
            self.weights[feature] = self.weights[feature] + (a * diff * feature_list[feature])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            for w in self.weights:
                print w[0]
            pass
