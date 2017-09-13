
import math
from colors import *
from sensors import *
from shapes import *
from robot import robot
import numpy as np
# know the

class environnement:
	def __init__(self, eSz, margin = 10, renderer = None):
		self.environnementSize = eSz
		self.margin = margin
		self.listOfObjects = []
		self.bounds = rectangle( eSz[0]/2,eSz[1]/2, eSz[0]-self.margin*2, eSz[1]-self.margin*2, 0, RED)
		self.addObject(self.bounds)

		self.renderer = renderer

	def addObject(self,obj):
		self.listOfObjects.append(obj)

	def removeObject(self,obj):
		self.listOfObjects.remove(obj)

	def clean(self):
		if self.renderer:
			self.renderer.clean()

	def draw(self):
		if self.renderer:
			for l in self.listOfObjects:
				l.draw(self.renderer)

	# TODO: Here we could cache the singleSegments avoiding the cost of
	# recomputation, We should flag a change and compute is changed
	def getSingleSegments(self):
		singleSegments = []
		#singleSegments = np.zeros(0)
		for index,obj in enumerate(self.listOfObjects):
			if isinstance(obj, basePolygon):
				nPoints = len(obj.dispPoints)
				#for i in range(nPoints):

				singleSegments.extend([[obj.dispPoints[i][0],obj.dispPoints[i][1],obj.dispPoints[0][0],obj.dispPoints[0][1],obj.color] if (i  == (nPoints-1)) else [obj.dispPoints[i][0],obj.dispPoints[i][1],obj.dispPoints[i+1][0],obj.dispPoints[i+1][1],obj.color] for i in range(nPoints)])

				"""
				if i == (nPoints-1):
					singleSegments.append([obj.dispPoints[i][0],obj.dispPoints[i][1],obj.dispPoints[0][0],obj.dispPoints[0][1],obj.color])
				else:
					singleSegments.append([obj.dispPoints[i][0],obj.dispPoints[i][1],obj.dispPoints[i+1][0],obj.dispPoints[i+1][1],obj.color])
				"""
		return singleSegments

	"""
	@profile
	def getSingleSegments(self):
		self.singleSegments = []
		for obj in self.listOfObjects:
			if isinstance(obj, basePolygon):
				nPoints = len(obj.dispPoints)
				for i in range(nPoints):
					if i == (nPoints-1):
						p1 = obj.dispPoints[i][0]
						p2 = obj.dispPoints[i][1]
						p3 = obj.dispPoints[0][0]
						p4 = obj.dispPoints[0][1]
					else:
						p1 = obj.dispPoints[i][0]
						p2 = obj.dispPoints[i][1]
						p3 = obj.dispPoints[i+1][0]
						p4 = obj.dispPoints[i+1][1]
					t = obj.color
					self.singleSegments.append([p1,p2,p3,p4,t])
		return self.singleSegments
	"""

	# Could make a unique
	def getSensors(self):
		l = []
		for obj in self.listOfObjects:
			if isinstance(obj, organ):
				l.append(obj)
		return l

	def getMouths(self):
		l = []
		for obj in self.listOfObjects:
			if isinstance(obj, mouth):
				l.append(obj)
		return l

	def getEars(self):
		l = []
		for obj in self.listOfObjects:
			if isinstance(obj, ear):
				l.append(obj)
		return l

	def getEyes(self):
		l = []
		for obj in self.listOfObjects:
			if isinstance(obj, eye):
				l.append(obj)
		return l

	def getFoods(self):
		l = []
		for obj in self.listOfObjects:
			if isinstance(obj, food):
				l.append(obj)
		return l

	def getRobots(self):
		l = []
		for obj in self.listOfObjects:
			if isinstance(obj, robot):
				l.append(obj)
		return l

	def getObjects(self):
		return self.listOfObjects

	def castRays(self,eye):
		if self.renderer:
			for obj in self.listOfObjects:
				for point in obj.displPoints:
					if eye.isPointInFov(point[0],point[1]):
						self.renderer.drawLine(BLUE,[eye.x, eye.y], point, 1)
					else:
						self.renderer.drawLine(RED,[eye.x, eye.y], point, 1)

	#@profile
	def intersectLines(self, pt1, pt2, ptA, ptB ):
	    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

	        returns a tuple: (xi, yi, valid, r, s), where
	        (xi, yi) is the intersection
	        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
	        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
	            valid == 0 if there are 0 or inf. intersections (invalid)
	            valid == 1 if it has a unique intersection ON the segment    """

	    DET_TOLERANCE = 0.00000001

	    # the first line is pt1 + r*(pt2-pt1)
	    # in component form:
	    x1, y1 = pt1;   x2, y2 = pt2
	    dx1 = x2 - x1;  dy1 = y2 - y1

	    # the second line is ptA + s*(ptB-ptA)
	    x, y = ptA;   xB, yB = ptB;
	    dx = xB - x;  dy = yB - y;

	    # we need to find the (typically unique) values of r and s
	    # that will satisfy
	    #
	    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
	    #
	    # which is the same as
	    #
	    #    [ dx1  -dx ][ r ] = [ x-x1 ]
	    #    [ dy1  -dy ][ s ] = [ y-y1 ]
	    #
	    # whose solution is
	    #
	    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
	    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
	    #
	    # where DET = (-dx1 * dy + dy1 * dx)
	    #
	    # if DET is too small, they're parallel
	    #
	    DET = (-dx1 * dy + dy1 * dx)

	    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

	    # now, the determinant should be OK
	    DETinv = 1.0/DET

	    # find the scalar amount along the "self" segment
	    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

	    # find the scalar amount along the input line
	    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

	    # return the average of the two descriptions
	    xi = (x1 + r*dx1 + x + s*dx)/2.0
	    yi = (y1 + r*dy1 + y + s*dy)/2.0
	    return ( xi, yi, 1, r, s )

class food(square):
	def __init__(self, x, y):
		square.__init__(self, x, y, 10.0, 0.0, BLUE)
		self.activateBBox()
		self.eatable = True
