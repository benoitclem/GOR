#! /usr/local/bin/python
# -*- coding: utf-8 -*-

import pickle
import os
from time import time
from GORLibrary	import game as GORGame
from GORLibrary import colors
#from GORLibrary import display as GORDisplay
from neat import Config, nn, population, statistics, parallel
from neat import DefaultGenome, DefaultReproduction
from neat import DefaultSpeciesSet, DefaultStagnation
import datetime

maxLightDist = 1000

class player(object):
	def __init__(self,renderer):
		self.renderer = renderer

class nnPlayer(player):
	def __init__(self,genome,config,renderer):
		player.__init__(self,renderer)
		print("I'm a machine player");
		self.genome = genome
		self.config = config
		self.net = nn.FeedForwardNetwork.create(self.genome,self.config)

	def play(self,inputs):
		## this is the what the robot sees ear

		run = True

		k = 0
		flatInput = []
		((img1,img2),life) = inputs

		# Just Display what the robot sees
		if img1:
			for i in range(len(img1)):
				angle = img1[len(img1)-1-i][0]
				dist = img1[len(img1)-1-i][1]
				col = img1[len(img1)-1-i][2]
				if dist>maxLightDist:
					dist = maxLightDist
				dist = ((maxLightDist - dist)/maxLightDist)
				angle = angle / 90.0
				#print(angle,col[0],col[1],col[2])
				r = col[0]*angle*dist;
				g = col[1]*angle*dist
				b = col[2]*angle*dist
				color = (r,g,b)
				flatInput.append((r/127.0)-1.0)
				flatInput.append((g/127.0)-1.0)
				flatInput.append((b/127.0)-1.0)
				#print(color)
				if self.renderer:
					for j in range(20):
						self.renderer.drawPixel(color,(k,j))
				k+=1

		if self.renderer:
			self.renderer.drawText((0,254,254),( k + 10 , 10),"%f"%(life))
			(di,da,run,exe,nPl) = self.renderer.getKeyboardInput()

		output = self.net.activate(flatInput)
		#print(output)
		return (output[0],output[1],False,run)

class humanPlayer(player):
	def __init__(self,renderer):
		player.__init__(self,renderer)
		print("I'm a human player, that ironic isn'it?")
		self.di = 0.00
		self.da = 0.00

	def play(self,inputs):
		run = True
		exe = True
		nPl = False

		k = 0

		((img1,img2),life) = inputs

		if img1:
			for i in range(len(img1)):
				angle = img1[i][0]
				dist = img1[i][1]
				col = img1[i][2]
				if dist>maxLightDist:
					dist = maxLightDist
				dist = ((maxLightDist - dist)/maxLightDist)
				angle = angle / 90.0
				#print(angle,col[0],col[1],col[2])
				color = (col[0]*angle*dist,col[1]*angle*dist,col[2]*angle*dist)
				if self.renderer:
					for j in range(20):
						self.renderer.drawPixel(color,(k,j))
				k+=1

		k+=10

		if img2:
			for i in range(len(img2)):
				angle = img2[i][0]
				dist = img2[i][1]
				col = img2[i][2]
				if dist>maxLightDist:
					dist = maxLightDist
				dist = ((maxLightDist - dist)/maxLightDist)
				angle = angle / 90.0
				#print(angle,col[0],col[1],col[2])
				color = (col[0]*angle*dist,col[1]*angle*dist,col[2]*angle*dist)
				if self.renderer:
					for j in range(20):
						self.renderer.drawPixel(color,(k,j))
				k+=1


		if self.renderer:
			self.renderer.drawText((0,254,254),( k + 10 , 10),"%f"%(life))
			(self.di,self.da,run,exe,npl) = self.renderer.getKeyboardInput()
		else:
			(self.di,self.da,run,exe,npl) = (0,0,True,True,False)

		return (self.di,self.da,nPl,run)

def checkExistanceOrCreate(fileName):
	if not os.path.exists(os.path.dirname(fileName)):
	    try:
	        os.makedirs(os.path.dirname(fileName))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	            raise

def recordGenomeIfNeeded(g,genIndex,iIndex):
	global batchName
	global recGenValue
	if recGenValue:
		if recGenValue > g.fitness:
			now = datetime.datetime.now()
			tmstp = now.strftime("%Y%m%d-%H%M")
			fileName = './%s/%03d/%03d-%f-%s.data' %(batchName,genIndex,iIndex,g.fitness*10000,tmstp)
			checkExistanceOrCreate(fileName)
			print("Record to %s" %(fileName))
			with open(fileName, 'wb') as f:
				pickle.dump(g, f)
		else:
			print(g.fitness)

def parallelEvalFitness(g):
	global configObj
	game = GORGame.GOR(740,580,10,None,None)
	game.addRobot(20,200)
	game.addFood()
	p = nnPlayer(g,configObj,None)
	game.setPlayer(p)
	g.fitness = game.run()
	print(g.fitness)

def evalFitness(genomes,configObj):
	global game
	global renderer
	indIndex = 0
	for genome_id, genome in genomes:
		p = nnPlayer(genome,configObj,renderer)
		#p = humanPlayer(renderer)
		game.setPlayer(p)
		genome.fitness = game.run()
		recordGenomeIfNeeded(genome,genome_id,indIndex)
		indIndex += 1

"""
from GORLibrary import display as GORDisplay
renderer = GORDisplay.pygameRenderer()
game = GORGame.GOR(740,580,10,humanPlayer(renderer),renderer)
game.addRobot(20,50)
for i in range(1):
	game.addFood()
game.run()
"""

pop = None
configName = raw_input("ConfigName? (Default:GorNnConfig)")
if configName == "":
	configName = "GorNnConfig"
configObj = Config(DefaultGenome,DefaultReproduction,
			DefaultSpeciesSet,DefaultStagnation,configName)
batchName = raw_input("BatchName? ")
if batchName == "":
	batchName = "unnamedTest"
recGenValue = 10
parallelExec = raw_input("Parallel Evaluation? (y/N) ")
if parallelExec == "":
	parallelExec = "n"
if parallelExec == 'y':
	print("Go for Threading stuffs")
	nThread = int(raw_input("N Threads? "))
	# The population stuff
	pop = population.Population(configObj)
	pe = parallel.ParallelEvaluator(nThread, parallelEvalFitness)
	pop.run(pe.evaluate, 300)

else:
	print("Go for single Thread")
	viz = raw_input("Visulisation (Y/n) ")
	if viz == '':
		viz = 'y'
	renderer = None
	if viz == 'y':
		from GORLibrary import display as GORDisplay
		renderer = GORDisplay.pygameRenderer()
	game = GORGame.GOR(740,580,10,None,renderer)
	game.addRobot(20,200)
	for i in range(1):
		game.addFood()

	#game.run()

	pop = population.Population(configObj)
	pop.run(evalFitness, 300)


# Save the winner.
print('Number of evaluations: {0:d}'.format(pop.total_evaluations))
winner = pop.most_fit_genomes[-1]
with open('nn_winner_genome', 'wb') as f:
    pickle.dump(winner, f)

print(winner)

# Plot the evolution of the best/average fitness.
# visualize.plot_stats(pop, ylog=True, filename="nn_fitness.svg")
# Visualizes speciation
# visualize.plot_species(pop, filename="nn_speciation.svg")

print("done")
