# -*- coding: utf-8 -*-

import pickle
from time import time
from GORLibrary	import game,colors
#from GORLibrary import display as GORDisplay
from neat import nn, population, statistics, visualize, parallel

maxLightDist = 1000

class player(object):
	def __init__(self,renderer):
		self.renderer = renderer

class nnPlayer(player):
	def __init__(self,genome,renderer):
		player.__init__(self,renderer)
		print("I'm a machine player");
		self.genome = genome
		self.net = nn.create_feed_forward_phenotype(self.genome)

	def play(self,inputs):
		## this is the what the robot sees ear

		k = 0
		flatInput = []
		((img1,img2),life) = inputs

		for i in range(len(img1)):
			angle = img1[i][0]
			dist = img1[i][1]
			col = img1[i][2]
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

			if self.renderer:
				for j in range(20):
					self.renderer.drawPixel(color,(k,j))
			k+=1

		if self.renderer:
			self.renderer.drawText(colors.YELLOW,( k + 10 , 10),"%f"%(life))
		#print(len(flatInput))
		#print(flatInput)
		output = self.net.serial_activate(flatInput)
		#print(output)
		return (output[0],output[1],False,True)

class humanPlayer(player):
	def __init__(self):
		player.__init__(self,renderer)
		print("I'm a human player, that ironic isn'it?")
		self.di = 0.00
		self.da = 0.00		

	def play(self,screen,inputs):
		run = True
		exe = True
		nPl = False

		## this is the what the robot sees ear

		k = 0
		((img1,img2),life) = inputs

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
			self.renderer.drawText(color,( k + 10 , 10),"%f"%(life))
			(self.di,self.da,run,exe,npl) = self.renderer.getKeyboardInput()
		else:
			(self.di,self.da,run,exe,npl) = (0,0,True,True,False)

		return (self.di,self.da,nPl,run)

def parallelEvalFitness(g):
	global game
	p = nnPlayer(g,None)
	game.setPlayer(p)
	g.fitness = game.run()
	print(g.fitness)

def evalFitness(gs):
	global game 
	global renderer
	for g in gs:
		p = nnPlayer(g,renderer)
		game.setPlayer(p)
		g.fitness = game.run()
		print(g.fitness)

pop = None
parallelExec = raw_input("Parallel Evaluation? (y/n) ")
if parallelExec == 'y':
	print("Go for Threading stuffs")
	nThread = int(raw_input("N Threads? "))
	# The population stuff
	pop = population.Population('GorNnConfig')
	pe = parallel.ParallelEvaluator(nThread, parallelEvalFitness)
	pop.run(pe.evaluate, 300)

else:
	print("Go for single Thread")
	viz = raw_input("Visulisation (y/n) ")
	renderer = None
	if viz == 'y':
		from GORLibrary import display as GORDisplay
		renderer = GORDisplay.pygameRenderer()
	game = game.GOR(740,580,10,None,renderer)
	game.addRobot(20,200)
	game.addFood()

	pop = population.Population('GorNnConfig')
	pop.run(evalFitness, 300)


# Save the winner.
print('Number of evaluations: {0:d}'.format(pop.total_evaluations))
winner = pop.most_fit_genomes[-1]
with open('nn_winner_genome', 'wb') as f:
    pickle.dump(winner, f)

print(winner)

# Plot the evolution of the best/average fitness.
visualize.plot_stats(pop, ylog=True, filename="nn_fitness.svg")
# Visualizes speciation
visualize.plot_species(pop, filename="nn_speciation.svg")

print("done")
