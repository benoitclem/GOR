
from random import random
from time import time, sleep
from GORLibrary import environnement
from GORLibrary import robot

class GOR:
	def __init__(self,szX,szY,margin,cb = None, renderer = None):

		# This is a hook to the player (human or neural-network)
		self.playerCb = cb

		# Scene and game params
		self.margin = margin
		self.screenW = szX+margin*2
		self.screenH = szY+margin*2
		self.screenSize = (self.screenW, self.screenH)
		self.nFood = 0;
		self.running = True

		self.renderer = renderer
		if self.renderer:
			self.renderer.init(self.screenSize)

		# Environnement Init
		self.env = environnement.environnement(self.screenSize,10,self.renderer)
		#sq = environnement.square(screenW/2,screenH/2,20,10,BLUE)
		#self.env.addObject(sq)

	def setPlayer(self,cb):
		self.playerCb = cb

	def isRunning(self):
		return self.running

	def quit(self):
		if self.renderer:
			self.rederer.quit()

	def randomLocation(self):
		x = int(self.margin+(random()*(self.screenW-(2*self.margin))))
		y = int(self.margin+(random()*(self.screenH-(2*self.margin))))
		return((x,y))

	def randomOrientaton(self):
		return random()*360

	def addRobot(self,sz,life = 10000):
		# Player
		(x,y) = self.randomLocation()
		orientation = self.randomOrientaton()
		self.robot = robot.robot(self.env,x,y,sz,90,life)
		self.env.addObject(self.robot)
		return self.robot

	def addFood(self):
		(x,y) = self.randomLocation()
		self.env.addObject(environnement.food(x,y))
		self.nFood += 1

	def removeFood(self,obj):
		self.env.removeObject(obj)
		self.nFood += 1

	def run(self):
		tStart = time()
		lifeRounds = 0
		fitness = 0.0
		self.running = True
		(x,y) = self.randomLocation()
		teta = self.randomOrientaton()
		self.robot.setAbsolutePosition(teta,x,y)
		self.robot.reanimate()
		while self.running:
			fitness += 0.001
			# Do the display
			if self.renderer:
				self.env.clean()
				self.env.draw()

			# Do the robot live
			for rb in self.env.getRobots():
				if self.playerCb:
					life = self.robot.life()
					imgs = rb.see()
					lifeRounds += 1
					(di,da,npl,run) = self.playerCb.play([imgs,life])

					self.running = run
					rb.move(da,di)
					for fd in self.env.getFoods():
						if rb.eat(fd):
							# we could relocate food instead of removing + addnew
							self.removeFood(fd)
							self.addFood()
				# Make the robot live
				self.robot.live()
				# Check if robot still alive
				if self.robot.isNotAlive():
					self.running = False

			# Commit display
			if self.renderer:
				#print("commiting")
				self.renderer.drawFps((0,254,254),(self.screenW-50,10))
				self.renderer.commit()
		tDelta = time() - tStart
		print("efficiency", tDelta/lifeRounds)
		return fitness
