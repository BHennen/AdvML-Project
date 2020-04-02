import pyglet
from pyglet.gl import *
import numpy as np
from ctypes import byref
import math

PIXEL_WIDTH = 96
PIXEL_HEIGHT = 96
VIEWER_WIDTH = 300
VIEWER_HEIGHT = 300
FPS = 50

VIEWERS = []

#TODO make code cleaner

def _nearest_pow2(v):
    # From http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    # Credit: Sean Anderson
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1

def update(dt):
    for viewer in VIEWERS:
        viewer.update(dt)

class PixelData():
    def __init__(self, width=PIXEL_WIDTH, height=PIXEL_HEIGHT, xpos=0, ypos=0, scale=1):
        # x = width/2
        # y = height/2
        self.xpos = xpos
        self.ypos = ypos
        self.width = width
        self.height = height
        self.scale = scale
        self.vlist = pyglet.graphics.vertex_list_indexed(4, [0, 0, 1, 2, 3, 3], ('v2f', [0,0, width,0, 0,height, width,height]),
                                                                                ('t2f', [0,0, 1,0, 0,1, 1,1]))
        self.alpha = np.full(shape = (width, height, 1), fill_value = 255, dtype = "uint8")
        self.pixels = np.zeros(shape=(width, height, 3), dtype = "uint8")

        self.pixels[:3, :, 0] = 255
        self.pixels[-3:, :, 0] = 255
        self.pixels[:, :3, 0] = 255
        self.pixels[:, -3:, 0] = 255
        self.pixels = np.concatenate((self.pixels, self.alpha), axis = 2)
        self.max_size = 2*scale
        self.min_size = scale
        self.cur_time = 0

    def update(self, dt):
        self.xpos += dt*1
        self.cur_time += dt
        self.scale = self.min_size + 1 + math.sin(self.cur_time) * (self.max_size - self.min_size)/2
        self.pixels = np.concatenate((np.random.randint(low = 0, high = 255, size = (self.width, self.height, 3), dtype = "uint8"), self.alpha), axis = 2)

    def create_empty_texture(self, width, height):
        # Create an empty texture to the nearest power of 2 and returns the texture height and width
        tex_id = GLuint()
        target = GL_TEXTURE_2D
        glGenTextures(1, byref(tex_id))
        glBindTexture(target, tex_id.value)
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        texture_width = _nearest_pow2(width)
        texture_height = _nearest_pow2(height)
        blank = (GLubyte * (texture_width * texture_height * 4))()
        glTexImage2D(target, 0,
                     GL_RGBA,
                     texture_width, texture_height,
                     0,
                     GL_RGBA, GL_UNSIGNED_BYTE,
                     blank)
        glFlush()
        return tex_id, texture_width, texture_height

    def render(self):
        pixels = np.flip(self.pixels, axis=0).flatten()
        tex_data = (GLubyte * pixels.size)( *pixels.astype('uint8'))
        glPushMatrix()
        glTranslatef(self.xpos, self.ypos, 0)
        glScalef(self.scale, self.scale, self.scale)
        target = GL_TEXTURE_2D
        glEnable(target)
        if self.width & 0x1:
            alignment = 1
        elif self.width & 0x2:
            alignment = 2
        else:
            alignment = 4
        glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT)
        glPixelStorei(GL_UNPACK_ALIGNMENT, alignment)
        self.create_empty_texture(self.width, self.height)
        glTexSubImage2D(target,
                        0,
                        1,
                        1,
                        self.width,
                        self.height,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        tex_data)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data)
        self.vlist.draw(GL_TRIANGLE_STRIP)
        glPopClientAttrib()
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        # # Working
        # pixels = np.flip(self.pixels, axis=0).flatten()
        # tex_data = (GLubyte * pixels.size)( *pixels.astype('uint8'))
        # glRasterPos2d(0, 0)
        # glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, tex_data)


class Viewer():
    def __init__(self, width=VIEWER_WIDTH, height=VIEWER_HEIGHT):
        self.width = width
        self.height = height
        self.objs = []
        self.window = pyglet.window.Window(width, height)
        self.window.push_handlers(self.on_draw)

    def on_draw(self):
        self.window.switch_to()
        self.window.clear()
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        for obj in self.objs:
            obj.render()

    def add_obj(self, obj):
        self.objs.append(obj)
    
    def update(self, dt):
        for obj in self.objs:
            obj.update(dt)

if __name__ == "__main__":

    pixels = PixelData()

    win1 = Viewer()
    win1.add_obj(pixels)

    win2 = Viewer()
    win2.add_obj(pixels)

    VIEWERS.extend([win1, win2])

    pyglet.clock.schedule_interval(update, 1.0/FPS)
    pyglet.app.run()

