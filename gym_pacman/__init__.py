#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
VERSION = "1.01"

from gym_interface import *
import gym

env_name = ''

def unregister(env_name):
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

unregister('GymPacman-v0')
gym.envs.registration.register(
    id='GymPacman-v0',
    entry_point='gym_pacman:GymPacman',
    max_episode_steps=0
)

unregister('GymPacmanRaw-v0')
gym.envs.registration.register(
    id='GymPacmanRaw-v0',
    entry_point='gym_pacman:GymPacman',
    max_episode_steps=0,
    kwargs=dict(obsType='raw')
)