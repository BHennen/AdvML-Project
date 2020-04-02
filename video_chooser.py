import pyglet
from pyglet import gl
import numpy as np


WIDTH = 200
HEIGHT = 200
FPS = 50

VIEWERS = []

def update(dt):
    for viewer in VIEWERS:
        viewer.update(dt)

class PixelData():
    def __init__(self):
        self.update(0)

    def update(self, dt):
        self.pixels = np.random.randint(low = 0, high = 255, size = (WIDTH, HEIGHT, 3), dtype = "uint8")

    def render(self):
        # Working
        pixels = np.flip(self.pixels, axis=0).flatten()
        tex_data = (gl.GLubyte * pixels.size)( *pixels.astype('uint8'))
        gl.glRasterPos2d(0, 0)
        gl.glDrawPixels(WIDTH, HEIGHT, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, tex_data)


class Viewer():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.objs = []
        self.window = pyglet.window.Window(width, height)
        self.window.push_handlers(self.on_draw)

    def on_draw(self):
        self.window.switch_to()
        self.window.clear()
        for obj in self.objs:
            obj.render()

    def add_obj(self, obj):
        self.objs.append(obj)
    
    def update(self, dt):
        for obj in self.objs:
            obj.update(dt)

if __name__ == "__main__":

    pixels = PixelData()
    
    win1 = Viewer(WIDTH, HEIGHT)
    win1.add_obj(pixels)

    win2 = Viewer(WIDTH, HEIGHT)
    win2.add_obj(pixels)

    VIEWERS.extend([win1, win2])

    pyglet.clock.schedule_interval(update, 1.0/FPS)
    pyglet.app.run()

