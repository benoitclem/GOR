
import pygame
from pygame.locals import *
from colors import *

class display:

	def init(self,screensize):
		pass

	def quit(self):
		pass

	def commit(self):
		pass

	def clean(self):
		pass

	def drawLine(self,COLOR,p1,p2,w = 1):
		pass

	def drawLines(self,COLOR,pts,w = 1):
		pass

	def drawCircle(self,COLOR,c,r,w):
		pass

	def drawArc(self,COLOR,bounds,p1,p2,w):
		pass

	def drawPixel(self,COLOR,p):
		pass

	def drawText(self,COLOR,p,text):
		pass

	def getKeyboardInput(self):
		return (0,0,True,True,False)

class pygameRenderer(display):

	def init(self,screenSize):

		self.lastDi = 0
		self.lastDa = 0

		self.screenSize = screenSize
		# Scene Init
		pygame.init()
		self.fpsClock = pygame.time.Clock()
		self.screen = pygame.display.set_mode(self.screenSize)

		self.font = pygame.font.SysFont("monospace", 15)

	def quit(self):
		pygame.quit()

	def commit(self):
		pygame.display.flip()

	def clean(self):
		self.screen.fill(BLACK)

	def drawLine(self,COLOR,p1,p2,w = 1):
		pygame.draw.line(self.screen, COLOR, p1, p2, w)

	def drawLines(self,COLOR,pts,w = 1):
		pygame.draw.lines(self.screen,COLOR,True,pts,w)

	def drawCircle(self,COLOR,c,r,w):
		pygame.draw.circle(self.screen,COLOR,c,r,w)

	def drawArc(self,COLOR,bounds,p1,p2,w):
		pygame.draw.arc(self.screen, COLOR, bounds, p1 , p2, w)

	def drawPixel(self,COLOR,p):
		self.screen.set_at(p, COLOR)

	def drawText(self,COLOR,p,text):
		self.screen.blit(self.font.render(text, 1, COLOR), p)

	def drawFps(self,COLOR,p):
		self.fpsClock.tick()
		self.drawText(COLOR,p,"%.1f fps"%(self.fpsClock.get_fps()))

	def getKeyboardInput(self):
		da = self.lastDa
		di = self.lastDi
		run = True # event to quit app
		exe = True # event to block exec
		nPl = False # event to add new player
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					exe = False
				elif (event.key == pygame.K_z) or (event.key == pygame.K_UP):
					di = 1.0
				elif (event.key == pygame.K_s) or (event.key == pygame.K_DOWN):
					di = -1.0
				elif (event.key == pygame.K_d) or (event.key == pygame.K_RIGHT):
					da = -1.0
				elif (event.key == pygame.K_q) or (event.key == pygame.K_LEFT):
					da = 1.0
				elif (event.key == pygame.K_n):
					nPl = True

			if event.type == pygame.KEYUP:
				if event.key == pygame.K_SPACE:
					exe = True
				elif (event.key == pygame.K_z) or (event.key == pygame.K_UP):
					di = 0.0
				elif (event.key == pygame.K_s) or (event.key == pygame.K_DOWN):
					di = 0.0
				elif (event.key == pygame.K_d) or (event.key == pygame.K_RIGHT):
					da = 0.0
				elif (event.key == pygame.K_q) or (event.key == pygame.K_LEFT):
					da = 0.0
				elif (event.key == pygame.K_n):
					nPl = False

			if event.type == QUIT:
				run = False

		self.lastDa = da
		self.lastDi = di
		return (di,da,run,exe,nPl)
