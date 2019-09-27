#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from gym_pacman.pacman import runGames, loadAgent, PacmanRules, Directions
from gym_pacman.layout import getLayout
from gym_pacman.game import Agent, GameOverSignal, Actions

import threading
import gym
from gym import spaces
import numpy as np

class GymPacAgentI(Agent):
    def __init__(self):
        super().__init__()
        
        self.actionEvent = threading.Event()
        self.observationEvent = threading.Event()
        self.exception = None
        
        self.lastScore = 0
        self.state = None
        self.reward = None
        self.action = None
        self.done = False
        
        self.gameOverSignal = False
                    
    def getAction(self, state):        
        self.reward = state.getScore() - self.lastScore
        self.lastScore = state.getScore() 
        self.state = state
        
        self.observationEvent.set()
        self.actionEvent.wait()
        
        if self.gameOverSignal:
            raise GameOverSignal()
                
        action = self.action
        self.actionEvent.clear()
        
        return action

    def final(self, state):
        self.state = state
        self.done = True
        self.observationEvent.set()       
        
class PacmanThread(threading.Thread):
    def __init__(self, agent, **args):
        super().__init__()
        self.args = args
        self.agent = agent
        self.games = None
    
    def run(self):
        # default args
        args = {
            'record': False,
            'catchExceptions': False,
            'numGames': 1,
            'timeout': 0.1,
            'numTraining': 0,
            'maxSteps': None,  # None
            'illegalAllowed': False
        }
        
        # fill in any matching values from self.args
        for k, v in args.items():
            v = self.args.get(k)
            if not v is None:
                args[k] = v

        # read params from self.args, or fill in the default values
        noKeyboard = True
        frameTime = self.args.get('frameTime', 0.1)
        quietGraphics = self.args.get('quietGraphics', False)
        textGraphics = self.args.get('textGraphics', False)
        numGhosts = self.args.get('numGhosts', None)
        zoom = self.args.get('zoom', 1)
        modulePath = self.args.get('modulePath', '')
        ghostTypeName = self.args.get('ghostType', 'RandomGhost')
        layout = self.args["layout"]
        args["layout"] = layout
        
        if numGhosts is None:
            numGhosts = args["layout"].getNumGhosts()

        if not 'pacman' in args:
            args['pacman'] = self.agent

        if not 'ghosts' in args:
            ghostType = loadAgent(ghostTypeName, noKeyboard, modulePath)
            args['ghosts'] = [ghostType(i+1) for i in range(numGhosts)]
        
        # Choose a display format
        if quietGraphics:
            import textDisplay
            args['display'] = textDisplay.NullGraphics()
        elif textGraphics:
            import textDisplay
            textDisplay.SLEEP_TIME = frameTime
            args['display'] = textDisplay.PacmanGraphics()
        else:
            import graphicsDisplay
            args['display'] = graphicsDisplay.PacmanGraphics(zoom, frameTime=frameTime)

        try:
            self.games = runGames(**args)
        except Exception as data:
            # make sure everything shuts down gracefully upon an exception
            self.agent.exception = data
            args['display'].finish()
            self.agent.observationEvent.set()

class StateMatExtractor:
    def __init__(self):
        self.num_feature_dims = 6
    
    def __call__(self, state):
        statemat = np.zeros((state.data.layout.width,
                             state.data.layout.height,
                             self.num_feature_dims))

        # pacman position
        pacman_pos = state.getPacmanPosition()
        statemat[int(pacman_pos[0]), int(pacman_pos[1]), 0] = True
        
        # ghosts
        for g in state.getGhostStates():
            pos = g.getPosition()

            # scared ghosts go under feature 5, normal ghosts under 1
            if g.scaredTimer > 0:
                statemat[int(pos[0]), int(pos[1]), 5] = True
            else:
                statemat[int(pos[0]), int(pos[1]), 1] = True

        # walls
        statemat[:, :, 2] = np.array(state.getWalls().data)
        # food
        statemat[:, :, 3] = np.array(state.getFood().data)     
        
        # capsules
        capsules = state.getCapsules()
        for c in capsules:
            statemat[c[0], c[1], 4] = True

        return statemat
        
class GymPacman(gym.Env):
    """
    A gym environment for the game of Pacman.
    
    Parameters:
        
    * obsType (ndarray, raw): Type of observations – either an array composed \
    of binary feature matrices, or the raw object representing the game state.
    * allowStop: Specifies whether 'Stop' is a legal action.
    * numericActions: Specifies whether actions are coded as strings or \
          numerically (['North', 'South', 'East', 'West', 'Stop']).
    * rewardShaping: Whether some basic reward shaping should be applied \
    to make the task more manageable.
    * layout: The name of the game layout.
    * layoutPath: Path to the layout files folder.
    * illegalAllowed: If True, illegal actions do not raise an exception.
    * quietGraphics: If True, graphics are suppressed.
    * textGraphics: If True, text graphics are used instead of the regular.
    """
    def __init__(self, obsType="ndarray", allowStop=False,
                 numericActions=True, rewardShaping=True,
                 illegalAllowed=True, quietGraphics=True,
                 **args):
        args.update(illegalAllowed=illegalAllowed,
                    quietGraphics=quietGraphics)
        self.agentI = None
        self.args = args
        self.pacmanThread = None
        self.action_space = spaces.Discrete(len(Actions._directions))
        self.observation_space = None
        self.obsType = obsType
        self.allowStop = allowStop
        self.numericActions = numericActions
        self.rewardShaping = rewardShaping
        
        # load the layout
        layout = self.args.get('layout', 'mediumClassic')
        layoutPath = self.args.get('layoutPath', '')
        self.layout = args['layout'] = getLayout(layout, layoutPath=layoutPath)

        if args['layout'] is None:
            raise Exception("The layout {} cannot be found".format(layout))
            
        # set up the observation space
        if self.obsType == "ndarray":
            self.extractor = StateMatExtractor()
            self.observation_space = spaces.Box(low=0, high=1,
                                    shape=(self.layout.width,
                                           self.layout.height,
                                    self.extractor.num_feature_dims))
        elif self.obsType == "raw":
            pass
        else:
            raise RuntimeError("Unknown observation type '{}'.".format(obsType))

    def __procState(self, state):
        if self.obsType == "raw":
            return state
        elif self.obsType == "ndarray":
            return self.extractor(state)
        else:
            raise RuntimeError("Unknown observation type '{}'.".format(self.obsType))

    def render(self, mode='human', close=False):
        pass

    def step(self, action):
        if self.agentI.done:
            raise Exception("The game is over. Call reset to start a new game.")

        info = {"won": False}
        
        try:
            if self.numericActions:
                action = self.actions[action]
        except:
            action = None

        self.agentI.action = action
        self.agentI.actionEvent.set()
        
        self.agentI.observationEvent.wait()
        
        won = self.agentI.state.isWin()
        state = self.__procState(self.agentI.state)
        reward = self.agentI.reward
        self.agentI.observationEvent.clear()
        
        # make sure everything shuts down gracefully upon an exception
        if not self.agentI.exception is None:
            self.agentI.done = True
            self.pacmanThread = None
            raise self.agentI.exception
        
        info["raw_reward"] = reward
        
        if self.rewardShaping:
            if reward > 20:
                reward = 50.0
            elif reward > 0:
                reward = 10.0
            elif reward < -10:
                reward = -500.0
                won = False
            elif reward < 0:
                reward = -1.0
                
        if won:
            reward = 100.0
            info["won"] = True
        
        return state, reward, self.agentI.done, info
    
    def getGameStats(self):
        """
        Once the game is over, the resulting game stats can be accessed using
        this. The object is cleared (replaced by None) once a new game is
        started using reset.
        """
        return self.pacmanThread.games[0]
        
    def getLegalActions(self, numeric=True):
        if self.agentI.done:
            return []

        legals = PacmanRules.getLegalActions(self.agentI.state)
        
        if not self.allowStop:
            legals.remove("Stop")
            
        if numeric:
            legals = [self.actionMap[l] for l in legals]
        
        return legals
        
    def close(self):
        if self.agentI:
            self.agentI.gameOverSignal = True
            self.agentI.actionEvent.set()
    
    def reset(self):
        if not self.pacmanThread is None:
            self.agentI.gameOverSignal = True
            self.agentI.actionEvent.set()
            self.pacmanThread.join()
        
        self.agentI = GymPacAgentI()
        self.pacmanThread = PacmanThread(self.agentI, **self.args)
        self.pacmanThread.start()
                
        self.agentI.observationEvent.wait()
        
        # make sure everything shuts down gracefully upon an exception
        if not self.agentI.exception is None:
            self.agentI.done = True
            self.pacmanThread = None
            raise self.agentI.exception
            
        state = self.__procState(self.agentI.state)

        self.actions = Directions.ALL_DIRECTIONS
        self.actionMap = {a: i for i, a in enumerate(self.actions)}
    
        self.agentI.observationEvent.clear()
        
        return state
        
    def __del__(self):
        self.close()
