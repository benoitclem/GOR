
from math import sqrt
from math import sin
from math import cos
from math import pi
from math import tan
from math import atan
from math import atan2
from colors import *

class basePolygon(object):
	def __init__(self,x,y,angle,color):
		object.__init__(self)
		self.x = x
		self.y = y
		self.orientation = angle
		self.color = color
		self.eatable = False
		self.bboxSize = 0.0
		self.bboxActive = False

	def cartToPolar(self,points):
		polPoints = []
		for point in points:
			dx = point[0]
			dy = point[1]
			teta = atan2(dy,dx)
			rho = sqrt(dx**2+dy**2)
			polPoints.append([teta*(180/pi),rho])
		return polPoints

	def polarToCart(self,points,angle):
		cartPoints = []
		for point in points:
			x = cos((angle + point[0])*(pi/180.0)) * point[1]
			y = sin((angle + point[0])*(pi/180.0)) * point[1]
			cartPoints.append([x,y])
		return cartPoints

	def polarToDisp(self,points,angle):
		dispPoints = []
		pts = self.polarToCart(points,angle)
		for pt in pts:
			dispPoints.append([self.x + pt[0], self.y - pt[1]])
		return dispPoints

	def rotate(self,angle):
		self.orientation += angle

	def orientate(self,orientation):
		self.orientation = orientation

	def setPos(self,pos):
		self.x = pos[0]
		self.y = pos[1]

	def update(self):
		self.dispPoints = self.polarToDisp(self.polarPoints,self.orientation)

	def draw(self,renderer = None):
		self.update()
		# bbox (circle)
		if renderer:
			if self.bboxActive:
				renderer.drawCircle(FUSHIA,(int(self.x),int(self.y)),self.bboxSize,1)
			renderer.drawLines(self.color,self.dispPoints,1)

	def activateBBox(self):
		self.bboxActive = True

	def disableBBox(self):
		self.bboxActive = False

	def computeBBoxSize(self):
		# bbox is computed with the more distant point from center
		for polarPoint in self.polarPoints:
			if polarPoint[1]>self.bboxSize:
				self.bboxSize = polarPoint[1]
		self.bboxSize = int(self.bboxSize)

class polygon(basePolygon):
	def __init__(self, x, y, points, color, angle):
		basePolygon.__init__(self, x, y, angle, color)
		# relative to origin of shape
		self.cartPoints = points
		self.polarPoints = self.cartToPolar(self.cartPoints)
		# absolute
		self.dispPoints = self.polarToDisp(self.polarPoints,self.orientation)
		self.computeBBoxSize()

class rectangle(basePolygon):
	def __init__(self, x, y, sizex, sizey, angle, color):
		basePolygon.__init__(self, x, y, angle, color)
		self.sizex = sizex
		self.sizey = sizey

		# relative to origin of shape
		ax = (self.sizex/2)
		ay = (self.sizey/2)
		bx = -(self.sizex/2)
		by = (self.sizey/2)
		cx = -(self.sizex/2)
		cy = -(self.sizey/2) 
		dx = (self.sizex/2)
		dy = -(self.sizey/2)		
		self.cartPoints = [[ax,ay],[bx,by],[cx,cy],[dx,dy]]
		# print(self.cartPoints)
		self.polarPoints = self.cartToPolar(self.cartPoints)
		# print(self.polarPoints)
		# absolute
		self.dispPoints = self.polarToDisp(self.polarPoints,self.orientation)
		self.computeBBoxSize()

class square(rectangle):
	def __init__(self, x,y, size, angle, color):
		rectangle.__init__(self, x, y, size, size, angle, color)
