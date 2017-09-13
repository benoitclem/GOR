# -*- coding: utf-8 -*-

from colors import *
from math import sqrt
from math import sin
from math import cos
from math import pi
from math import atan
from math import atan2
from time import time
import numpy as np

class organ(object):
	def __init__(self, environnement, angle, orientation, x, y):
		object.__init__(self)
		self.environnement = environnement
		self.angle = angle
		self.fovGuide = 40
		self.orientation = orientation
		self.orientationGuide = 60
		self.x = x
		self.y = y

	#def getColor(self):
	#	return BLACK

	def draw(self, renderer = None):
		if renderer:
			col = self.getColor()

			# Orientation
			orientX = self.x + cos(self.orientation*(pi/180.0)) * self.orientationGuide
			orientY = self.y - sin(self.orientation*(pi/180.0)) * self.orientationGuide
			renderer.drawLine(col, [self.x, self.y], [orientX,orientY], 1)

			# fov
			# L1
			firstA = (self.orientation-self.angle/2.0)*(pi/180.0)
			firstX = self.x + cos(firstA) * self.fovGuide
			firstY = self.y - sin(firstA) * self.fovGuide
			renderer.drawLine(col, [self.x, self.y], [firstX, firstY], 1)

			# L2
			secondA = (self.orientation+self.angle/2.0)*(pi/180.0)
			secondX = self.x + cos(secondA) * self.fovGuide
			secondY = self.y - sin(secondA) * self.fovGuide
			renderer.drawLine(col, [self.x, self.y], [secondX, secondY], 1)

			# Arc
			arcBounds = [self.x-self.fovGuide, self.y-self.fovGuide, 2*self.fovGuide, 2*self.fovGuide]
			renderer.drawArc(col, arcBounds, firstA , secondA, 1)

	def update(self,x,y,orientation):
		self.x = x
		self.y = y
		self.orientation = orientation

	def turn(self):
		#print(self.orientation)
		self.orientation += 1
		if self.orientation > 180:
			self.orientation = -179

	def isPointInFov(self, px, py):

		dx = float(max(px,self.x)-min(px,self.x))
		dy = float(max(py,self.y)-min(py,self.y))
		#print(dx,dy)
		if dx == 0:
			if dy == 0:
				return false
			else:
				teta = 90.0
		else:
			# Work in upper right quarter
			teta = atan(dy/dx) * (180 / pi)

		if self.x < px:
			pass
		else:
			teta = 180 - teta

		if py < self.y:
			pass
		else:
			teta = - teta

		#print(teta)
		bounds = []

		if self.orientation >= 0:
			upperBound = self.orientation + self.angle/2
			lowerBound = self.orientation - self.angle/2
			if upperBound > 180:
				secondUpperBound = - 180 + (upperBound - 180)
				upperBound = 180
				secondLowerBound = - 180
				bounds.append([secondUpperBound,secondLowerBound])
			bounds.append([upperBound,lowerBound])

		if self.orientation < 0:
			upperBound = self.orientation + self.angle/2
			lowerBound = self.orientation - self.angle/2
			if lowerBound < -180:
				secondLowerBound =  180 + (lowerBound + 180)
				lowerBound = - 180
				secondUpperBound =  180
				bounds.append([secondUpperBound,secondLowerBound])
			bounds.append([upperBound,lowerBound])

		visible = False

		#print('======')
		for bound in bounds:
			lowerBound = bound[1]
			upperBound = bound[0]
			#print('Low',lowerBound,'teta',teta,'up',upperBound)
			if (lowerBound<teta) and (teta<upperBound):
				visible = True

		return visible

class eye(organ):
	def __init__(self, environnement, angle, orientation, x, y, resolution):
		organ.__init__(self, environnement, angle, orientation, x, y)
		self.resolution = resolution # points par degrés
		self.angleList = np.linspace(-self.angle,self.angle,self.angle*self.resolution)
		self.nRays = len(self.angleList)

	def getColor(self):
		return GREEN

	def do(self):
		self.see()

	# ALL
	# 0.0017966909408569337 - 0.000198841512151069
	# 1.6ms
	# 0.0017966909408569337 - 0.0009033533949030144
	#@profile
	def see2(self):
		#t0 = time()
		visionBuffer = [[0,8000,(0,0,0)]] * self.nRays
		segments = self.environnement.getSingleSegments()
		#print("=======")
		for i,curDeg in enumerate(self.angleList):
		#while curDeg <= self.angle/2.0:
			for j,segment in enumerate(segments):
				# segment
				ax = float(segment[0])
				ay = float(segment[1])
				bx = float(segment[2])
				by = float(segment[3])
				color = segment[4]

				angle = (self.orientation+curDeg)*(pi/180.0)
				cx = float(self.x)
				cy = float(self.y)
				dx = float(self.x + (cos(angle) * 1000.0))
				dy = float(self.y - (sin(angle) * 1000.0))

				dxl1 = bx - ax
				dyl1 = by - ay

				dxl2 = dx - cx
				dyl2 = dy - cy

				#pygame.draw.line(screen, GREEN, [int(cx), int(cy)], [int(dx),int(dy)], 1)

				p = self.environnement.intersectLines((ax,ay),(bx,by),(cx,cy),(dx,dy))
				"""
				if p != None:
					if p[2] != 0:
						x = int(p[0])
						y = int(p[1])
						r = p[3]
						s = p[4]

						if (r<=1.0) and (r>=0.0) and (s<=1.0) and (s>=0):
							a1 = atan2(dyl1,dxl1) * (180.0 / pi)
							a2 = atan2(dyl2,dxl2) * (180.0 / pi)
							#print("before",a1,a2)
							if a1 > 90.0:
								a1 -= 180.0
							elif a1 < -90.0:
								a1 += 180.0

							if a2 > 90.0:
								a2 -= 180.0
							elif a2 < -90.0:
								a2 += 180.0
							#print("after",a1,a2)
							maxA = max(a1,a2)
							minA = min(a1,a2)
							teta = (maxA - minA)
							#print("angle",teta)
							if teta > 90.0:
								teta -= 2*(teta-90.0)
							#print("angle",teta)
							#compute distance
							angle = (teta/90.0) * 255.0

							# See if this intersection is near that the
							dist = sqrt((x-self.x)*(x-self.x) + (y-self.y)*(y-self.y))
							if dist<visionBuffer[i][2]:
								#nearestIntersection = (x,y,dist,angle,teta,a1,a2,color)
								visionBuffer[i] = [teta,dist,color]
							#pygame.draw.circle(screen,BLUE,(x,y),10)
				"""
		#print(time()-t0) # It takes about 2ms for 1 pix/deg
		return visionBuffer

	#@profile
	def see(self):
		visionBuffer = []
		#visionBuffer = [[0,0,0]] * self.nRays
		segments = self.environnement.getSingleSegments()
		dDeg = 1.0/self.resolution
		curDeg = -self.angle/2.0
		#print("=======")
		for i,curDeg in enumerate(self.angleList):
		#while curDeg <= self.angle/2.0:
			intersections = []
			for segment in segments:
				# segment
				ax = float(segment[0])
				ay = float(segment[1])
				bx = float(segment[2])
				by = float(segment[3])
				color = segment[4]

				angle = (self.orientation+curDeg)*(pi/180.0)
				cx = float(self.x)
				cy = float(self.y)
				dx = float(self.x + (cos(angle) * 1000.0))
				dy = float(self.y - (sin(angle) * 1000.0))

				dxl1 = bx - ax
				dyl1 = by - ay

				dxl2 = dx - cx
				dyl2 = dy - cy

				#pygame.draw.line(screen, GREEN, [int(cx), int(cy)], [int(dx),int(dy)], 1)
				p = self.environnement.intersectLines((ax,ay),(bx,by),(cx,cy),(dx,dy))
				if p != None:
					if p[2] != 0:
						x = int(p[0])
						y = int(p[1])
						r = p[3]
						s = p[4]

						if (r<=1.0) and (r>=0.0) and (s<=1.0) and (s>=0):
							a1 = atan2(dyl1,dxl1) * (180.0 / pi)
							a2 = atan2(dyl2,dxl2) * (180.0 / pi)
							#print("before",a1,a2)
							if a1 > 90.0:
								a1 -= 180.0
							elif a1 < -90.0:
								a1 += 180.0

							if a2 > 90.0:
								a2 -= 180.0
							elif a2 < -90.0:
								a2 += 180.0
							#print("after",a1,a2)
							maxA = max(a1,a2)
							minA = min(a1,a2)
							teta = (maxA - minA)
							#print("angle",teta)
							if teta > 90.0:
								teta -= 2*(teta-90.0)
							#print("angle",teta)
							#compute distance
							angle = (teta/90.0) * 255.0
							intersections.append((x,y,angle,teta,a1,a2,color))
							#pygame.draw.circle(screen,BLUE,(x,y),10)

			nearestIntersection = (0,0,8000,0,0,0,0,(0,0,0))
			for (x,y,angle,teta,a1,a2,color) in intersections:
				l = sqrt((x-self.x)*(x-self.x) + (y-self.y)*(y-self.y))
				if l<nearestIntersection[2]:
					nearestIntersection = (x,y,l,angle,teta,a1,a2,color)

			nx = nearestIntersection[0]
			ny = nearestIntersection[1]
			nDist = nearestIntersection[2]
			nAngle = nearestIntersection[3]
			nTeta = nearestIntersection[4]
			na1 = nearestIntersection[5]
			na2 = nearestIntersection[6]

			color = nearestIntersection[7]
			#pygame.draw.line(screen, (ncol,ncol,ncol), [int(self.x), int(self.y)], [int(nx),int(ny)], 1)
			#pygame.draw.circle(screen,BLUE,(nx,ny),2)
			visionBuffer.append([nTeta,nDist,color])
			#print(na1,na2,nTeta)
		return visionBuffer

class mouth(organ):
	def __init__(self, environnement, angle, orientation, x, y):
		organ.__init__(self, environnement, angle, orientation, x, y)
		self.value = 1000.0

	def getColor(self):
		return RED

	def do(self):
		self.talk()

	def talk(self):
		return self.value

class ear(organ):
	def __init__(self, environnement, angle, orientation, x, y):
		organ.__init__(self, environnement, angle, orientation, x, y)

	def getColor(self):
		return FUSHIA

	def do(self):
		self.ear()

	def ear(self,screen):
		sound = 0.0
		for mouth in self.environnement.getMouths():
			#pygame.draw.line(screen, GREEN, [int(mouth.x), int(mouth.y)], [int(self.x),int(self.y)], 1)
			if self.isPointInFov(mouth.x,mouth.y) and mouth.isPointInFov(self.x,self.y):
				dx = mouth.x-self.x
				dy = mouth.y-self.y
				#sound += mouth.talk() / (2*pi * sqrt(dx*dx + dy*dy))
				sound += mouth.talk()
				pygame.draw.line(screen, GREEN, [int(mouth.x), int(mouth.y)], [int(self.x),int(self.y)], 1)
			else:
				pygame.draw.line(screen, RED, [int(mouth.x), int(mouth.y)], [int(self.x),int(self.y)], 1)
		return sound
