# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action) - returns (nextState,prob)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for it in range(self.iterations):
            values_it_copy = self.values.copy() # Copy values to use in current iteration, see "batch" version description on website
            for s in self.mdp.getStates(): # Iterate through each state
                if self.mdp.isTerminal(s) == True: # Enters here, but I don't think anything happens. In debugging the possible actions were "illegal", wouldn't produce v beyond v = 0
                    PossibleActions = self.mdp.getPossibleActions(s)
                    for action in PossibleActions:
                        v = 0
                        for trans in self.mdp.getTransitionStatesAndProbs(s,action):
                            trans_state = transition[0]
                            trans_prob = transition[1]
                            trans_reward = self.mdp.getReward(s, action, trans_state)
                            print trans_state
                            trans_lambda = self.discount
                            v += trans_prob * (trans_reward + (trans_lambda * self.values[trans_state]))
                        values_it_copy[s] = v
                else:
                    PossibleActions = self.mdp.getPossibleActions(s) # gets legal or possible actions at state s
                    valuesForMax = util.Counter() # This creates a dictionary to keep values as they are calculated
                    for action in PossibleActions:
                        v = 0
                        for transition in self.mdp.getTransitionStatesAndProbs(s,action):
                            trans_state = transition[0]
                            trans_prob = transition[1]
                            trans_reward = self.mdp.getReward(s,action,trans_state)
                            trans_lambda = self.discount
                            v += trans_prob * (trans_reward + (trans_lambda * self.values[trans_state])) # This sums each transitional probability and state combination
                        valuesForMax[action] = v
                    values_it_copy[s] = valuesForMax[valuesForMax.argMax()] # argMax returns index, set within the array
            self.values = values_it_copy



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        Qv = 0
        for trans in self.mdp.getTransitionStatesAndProbs(state,action):
            trans_state = trans[0]
            trans_prob = trans[1]
            trans_reward = self.mdp.getReward(state,action,trans_state)
            trans_lambda = self.discount
            Qv += trans_prob * (trans_reward + (trans_lambda * self.values[trans_state]))
        return Qv


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        PossibleActions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state) == False: # I think this wouldn't run right because the last state has no possible actions, the index of 0 threw errors. If the state is terminal this will do nothing
            init_action = PossibleActions[0]
            init_Q = self.getQValue(state,init_action)
            for a in PossibleActions:
                Qv = self.getQValue(state,a)
                if Qv > init_Q:
                    init_Q = Qv
                    init_action = a
            return init_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
