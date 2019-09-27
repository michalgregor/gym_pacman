#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from gym_pacman import GymPacman

env = GymPacman(layout='santa_fe', numericActions=False,
                quietGraphics=False)
obs = env.reset()

env.step('East')
env.step('East')
env.step('East')

env.step('South')
env.step('South')
env.step('South')
env.step('South')
env.step('South')

env.step('East')
env.step('East')
env.step('East')
env.step('East')
env.step('East')
env.step('East')
