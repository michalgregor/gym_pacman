#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
VERSION = "1.001"

from gym_interface import *
import gym

env_name = 'GymPacman-v0'

if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]
    
gym.envs.registration.register(
    id=env_name,
    entry_point='gym_pacman:GymPacman',
    max_episode_steps=0,
    kwargs=dict(illegalAllowed=True)
)
