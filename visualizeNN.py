# -*- coding: utf-8 -*-

import sys
import pickle
from neat import nn, population, statistics, visualize, parallel

if len(sys.argv) >= 2:
	fName = sys.argv[1]
	print(fName)
	winner = None
	with open(fName, 'rb') as f:
		winner = pickle.load(f)
	visualize.draw_net(winner, view=True, filename=fName+".gv")
	print(winner)
else:
	print("No file to print")