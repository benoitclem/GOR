# -*- coding: utf-8 -*-

from colors import *
from math import sqrt
from math import sin
from math import cos
from math import pi
from math import tan
from math import atan
from math import atan2
from sensors import *
from shapes import square

class robot(object):
	def __init__(self,env,x,y,size,orientation,life=200.0):
		object.__init__(self)
		self.environnement = env
		self.size = size
		self.x = x
		self.y = y
		self.lifeValue = life
		self.orientation = orientation
		self.bboxSize = int((sqrt(2)*self.size)/2)
		self.eyeAOW = 45.0
		self.eyeRes = 0.5

		# Create sensors with dummy position (center of robot)
		self.eyeR = eye(self.environnement,self.eyeAOW,self.orientation,self.x,self.y,self.eyeRes)
		self.eyeL = eye(self.environnement,self.eyeAOW,self.orientation,self.x,self.y,self.eyeRes)
		self.mouth = mouth(self.environnement,45.0,self.orientation,self.x,self.y)
		self.earR = ear(self.environnement,45.0,self.orientation,self.x,self.y)
		self.earL = ear(self.environnement,45.0,self.orientation,self.x,self.y)

		self.mySensors = []
		self.mySensors.append(self.eyeR)
		self.mySensors.append(self.eyeL)
		self.mySensors.append(self.mouth)
		self.mySensors.append(self.earR)
		self.mySensors.append(self.earL)

		# Keep sensors relative position
		self.eyeAngle = 20.0
		self.posEyeR = (self.eyeAngle,sqrt((self.size/2)**2+(tan((self.eyeAngle)*(pi/180.0))*(self.size/2))**2),0.0)
		self.posEyeL = (-self.eyeAngle,sqrt((self.size/2)**2+(tan((-self.eyeAngle)*(pi/180.0))*(self.size/2))**2),0.0)
		self.posMouth = (0.0,self.size/2,0)
		self.posEarR = (90.0,self.size/2,90.0)
		self.posEarL = (-90.0,self.size/2,-90.0)

		self.body = square(self.x,self.y,size,self.orientation,RED)

	def reanimate(self,life = 200.0):
		self.lifeValue = life

	def getEyeParameters(self):
		return (self.eyeAOW,self.eyeRes)

	def life(self):
		return self.lifeValue

	def live(self):
		self.lifeValue -= 0.1

	def isNotAlive(self):
		return self.lifeValue<=0.0

	def setSensorPosition(self, sensor,relPos):
		dx = cos((relPos[0]+self.orientation)*(pi/180.0)) * relPos[1]
		dy = sin((relPos[0]+self.orientation)*(pi/180.0)) * relPos[1]
		sensor.update(self.x+dx,self.y-dy,self.orientation+relPos[2])

	def draw(self,renderer = None):
		if renderer:
			# bbox (circle)
			renderer.drawCircle(FUSHIA,(int(self.x),int(self.y)),self.bboxSize,1)
			# body
			self.body.draw(renderer)
			# sensors
			for sens in self.mySensors:
				sens.draw(renderer)
	
	def eat(self,obj):
		dist = sqrt((max(self.x,obj.x)-min(self.x,obj.x))**2 + (max(self.y,obj.y)-min(self.y,obj.y))**2)
		if(dist<=(self.bboxSize+obj.bboxSize)):
			if obj.eatable:
				self.lifeValue += 300
				print("miam! life=" + str(self.lifeValue))
				if (self.lifeValue > 10000):
					print "Robot too efficient has been killed by jealous crowd"
					self.lifeValue = 0
			else:
				print("beurk")
			return obj.eatable

	def see(self):
		img1 = self.eyeR.see()
		img2 = self.eyeL.see()
		return (img1,img2)

	def move(self, dangle, distance):
		dx = cos((dangle+self.orientation)*(pi/180.0)) * distance
		dy = -sin((dangle+self.orientation)*(pi/180.0)) * distance
		#print(((0.05*distance) + (0.025*dangle)))
		self.lifeValue -= ((0.05*abs(distance) + (0.025*abs(dangle))))
		self.setPosition(dangle,dx,dy)

	def setPosition(self, dangle, dx, dy):
		self.orientation += dangle
		self.x += dx
		self.y += dy
		# Contain the robot inside the environement
		if ((self.x - self.bboxSize) < self.environnement.margin):
			self.x = self.environnement.margin + self.bboxSize;
		if ((self.y - self.bboxSize) < self.environnement.margin):
			self.y = self.environnement.margin + self.bboxSize;
		if ((self.x + self.bboxSize) > (self.environnement.environnementSize[0] - self.environnement.margin)):
			self.x = self.environnement.environnementSize[0] - self.environnement.margin - self.bboxSize;
		if ((self.y + self.bboxSize) > (self.environnement.environnementSize[1] - self.environnement.margin)):
			self.y = self.environnement.environnementSize[1] - self.environnement.margin - self.bboxSize;
		self.updateSensors()

	def setAbsolutePosition(self, angle, x, y):
		self.orientation = angle
		self.x = x
		self.y = y
		# Contain the robot inside the environement
		if ((self.x - self.bboxSize) < self.environnement.margin):
			self.x = self.environnement.margin + self.bboxSize;
		if ((self.y - self.bboxSize) < self.environnement.margin):
			self.y = self.environnement.margin + self.bboxSize;
		if ((self.x + self.bboxSize) > (self.environnement.environnementSize[0] - self.environnement.margin)):
			self.x = self.environnement.environnementSize[0] - self.environnement.margin - self.bboxSize;
		if ((self.y + self.bboxSize) > (self.environnement.environnementSize[1] - self.environnement.margin)):
			self.y = self.environnement.environnementSize[1] - self.environnement.margin - self.bboxSize;
		self.updateSensors()

	def updateSensors(self):
		# Apply the sensor positon with their relative position
		self.setSensorPosition(self.eyeR,self.posEyeR)
		self.setSensorPosition(self.eyeL,self.posEyeL)
		self.setSensorPosition(self.mouth,self.posMouth)
		self.setSensorPosition(self.earR,self.posEarR)
		self.setSensorPosition(self.earL,self.posEarL)

		self.body.setPos((self.x,self.y))
		self.body.orientate(self.orientation)
