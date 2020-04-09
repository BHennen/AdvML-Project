import pyglet
from pyglet.gl import *
import numpy as np
from colorsys import hsv_to_rgb

from image_manager import ImageManager


def draw_rect(x, y, width, height):
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()


class Widget(object):
    def __init__(self, parent, x, y, width, height):
        self._window = None
        self.parent = parent
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.children = []
        if parent is not None:
            parent.add_child(self)

    def resize(self):        
        '''
        Called whenever widget's dimesions are changed,
        or whenever the window is resized (to update with parent's new dimensions).
        Updates the current widget's dimensions and that of its children
        '''
        self._resize()
        for child in self.children:
            child.resize()

    def render(self):
        '''
        Called for graphical updates. 
        '''
        self._render()
        for child in self.children:
            child.render()

    def update(self, dt):
        '''
        For updates based on a clock interval.
        '''
        self._update(dt)
        for child in self.children:
            child.update(dt)

    def _resize(self):
        raise NotImplementedError()

    def _render(self):
        raise NotImplementedError()

    def _update(self, dt):
        raise NotImplementedError()

    def get_pos(self):
        return self.x, self.y

    def get_size(self):
        return self.width, self.height

    def get_window(self):
        return self._window

    def _set_window(self, window):
        pass

    def set_window(self, window):
        self._window = window
        self._set_window(window)
        for child in self.children:
            child.set_window(window)

    window = property(get_window, set_window)

    def add_child(self, child):
        child.parent = self
        if self.window is not None:
            child.window = self.window
        self.children.append(child)
        child.resize()

    @staticmethod
    def scale(percentage, value):
        if percentage is None:
            return None
        elif abs(percentage) > 1:
            return percentage
        else:
            return percentage * value


class CenteredWidget(Widget):
    '''
    Widget that is centered relative to its parent
    '''
    def __init__(self, parent, width, height, x_offset=None, y_offset=None):
        self.set_pos(width, height, x_offset, y_offset)
        super().__init__(parent, 0, 0, width, height)

    def set_pos(self, width, height, x_offset=None, y_offset=None):
        self._set_width = width
        self._set_height = height
        self.x_offset = x_offset
        self.y_offset = y_offset
    
    def _resize(self):
        self.width = Widget.scale(self._set_width, self.parent.width)
        self.height = Widget.scale(self._set_height, self.parent.height)
        x_offset = Widget.scale(self.x_offset, (self.parent.width - self.width) / 2)
        y_offset = Widget.scale(self.y_offset, (self.parent.height - self.height) / 2)

        parent_center_x = self.parent.x + self.parent.width / 2
        parent_center_y = self.parent.y + self.parent.height / 2
        self.x = parent_center_x - self.width/2 + (x_offset or 0)
        self.y = parent_center_y - self.height/2 + (y_offset or 0)

    def _render(self):
        pass

    def _update(self, dt):
        pass

class RelativeWidget(Widget):
    '''
    Widget that can be positioned relatively to its parent
    '''
    def __init__(self, parent, left=None, right=None, top=None, bottom=None, width=None, height=None):
        '''
        If both top and bottom are set, height is determined automatically.
        If both left and right are set, width is determined automatically.
        '''
        self.set_pos(left=left, right=right, top=top, bottom=bottom, width=width, height=height)
        super().__init__(parent, 0, 0, 0, 0)

    def set_pos(self, left=None, right=None, top=None, bottom=None, width=None, height=None):
        self._left = left
        self._right = right
        self._top = top
        self._bottom = bottom
        self._set_width = width
        self._set_height = height

    def _resize(self):        
        left = self.scale(self._left, self.parent.width)
        right = self.scale(self._right, self.parent.width)
        top = self.scale(self._top, self.parent.height)
        bottom = self.scale(self._bottom, self.parent.height)
        width = self.scale(self._set_width, self.parent.width)
        height = self.scale(self._set_height, self.parent.height)

        if left is not None and right is not None:
            # left and right set, so constrain width to be from x to right side of parent
            self.x = left + self.parent.x
            self.width = self.parent.x + self.parent.width - self.x - right
        else:
            if width is None:
                raise ValueError("Width must be set.")
            self.width = width
            if left is not None:
                self.x = left + self.parent.x
            elif right is not None:
                self.x = self.parent.x + self.parent.width - self.width
            else:
                raise ValueError("Left or Right must be set.")
        
        if top is not None and bottom is not None:
            # top and bottom set, so constrain height to be from y to top side of parent
            self.y = bottom + self.parent.y
            self.height = self.parent.y + self.parent.height - self.y - top
        else:
            if height is None:
                raise ValueError("Height must be set.")
            self.height = height
            if bottom is not None:
                self.y = bottom + self.parent.y
            elif top is not None:
                self.y = self.parent.y + self.parent.height - self.height
            else:
                raise ValueError("Top or Bottom must be set.")

    def _render(self):
        pass

    def _update(self, dt):
        pass


class WindowWidget(Widget):
    def __init__(self, window_width, window_height):
        super().__init__(None, 0, 0, window_width, window_height)
        self.window = pyglet.window.Window(window_width, window_height, resizable=True)
        self.window.push_handlers(on_resize = self._on_resize, on_draw = self.render)

    def _resize(self):
        pass

    def _render(self):
        pass

    def _update(self, dt):
        pass

    def _on_resize(self, width, height):
        self.width = width
        self.height = height
        self.resize()


class Button(CenteredWidget, pyglet.event.EventDispatcher):
    #Based on pyglet example media player: https://github.com/pyglet/pyglet/blob/master/examples/media_player.py
    def __init__(self, parent, text, width=None, height=None, x_offset=None, y_offset=None, text_padding=0):
        self._set_width = width
        self._set_height = height
        self.text_padding = text_padding
        self._text = pyglet.text.Label(f'{text or ""}', anchor_x='center', anchor_y='center')
        self.width = (width or self._text.content_width) + text_padding * 2
        self.height = (height or self._text.content_height) + text_padding * 2
        super().__init__(parent, width=self.width, height=self.height, x_offset=x_offset, y_offset=y_offset)
        self.pressed = False

    def _render(self):
        if self.pressed:
            glColor3f(1, 0, 0)
        draw_rect(self.x, self.y, self.width, self.height)
        glColor3f(1, 1, 1)
        self.draw_label()

    def draw_label(self):
        self._text.x = int(self.x + self.width / 2)
        self._text.y = int(self.y + self.height / 2)
        self._text.draw()

    def set_text(self, text):
        self._text.text = text
        self.width = (self._set_width or self._text.content_width) + self.text_padding * 2
        self.height = (self._set_height or self._text.content_height) + self.text_padding * 2

    text = property(lambda self: self._text.text, set_text)

    def _set_window(self, window):
        window.push_handlers(self)

    def hit_test(self, x, y):
        return (self.x < x < self.x + self.width and
                self.y < y < self.y + self.height)

    def on_mouse_press(self, x, y, button, modifiers):
        self.pressed = self.hit_test(x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.pressed = self.hit_test(x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        if self.hit_test(x, y):
            self.dispatch_event('on_press', self)
        self.pressed = False

Button.register_event_type('on_press')


class PixelFrame(RelativeWidget):
    def __init__(self, parent, pixel_width, pixel_height, pixel_seq, FPS=50, left=None, right=None, top=None, bottom=None, width=None, height=None):
        super().__init__(parent, left=left, right=right, top=top, bottom=bottom, width=width, height=height)
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width
        self.mgr = ImageManager(img_height = pixel_height, img_width = pixel_width, max_texture_size=1024)
        self.set_pixels(pixel_seq)
        self.img_index = 0
        self.switch_time = 1/FPS 
        self.countdown = self.switch_time

    def set_pixels(self, pixel_seq):
        print("Loading pixels...", end="", flush=True)        
        for index, pixels in enumerate(pixel_seq):
            self.mgr.update_image(pixels, index)
        self.num_imgs = len(pixel_seq)
        print("done")

    def _render(self):
        self.mgr.draw_image(self.img_index, self.x, self.y, self.width, self.height)

    def _update(self, dt):
        # Cycle through the images in the image manager
        self.countdown -= dt
        # Skip some images if too much time has passed
        while self.countdown < 0:
            self.countdown += self.switch_time
            self.img_index = (self.img_index+1) % self.num_imgs

def gen_rainbow_pixels(num_pixels, pixel_width, pixel_height):
    # Generates a list of pixels which is a hue rainbow when viewed sequentially 
    zeros = np.zeros(shape=(pixel_height, pixel_width, 3), dtype = "uint8")
    alpha = np.full(shape = (pixel_height, pixel_width, 1), fill_value = 255, dtype = "uint8")
    pixel_list = []
    for pixel_num in range(num_pixels):
        pixels = np.concatenate((zeros, alpha), axis = 2)
        hue = pixel_num / num_pixels
        rgb = hsv_to_rgb(h=hue, s=1, v=1)
        colors = [col*255 for col in rgb] + [255]
        pixels[:1, :, :] = colors
        pixels[-1:, :, :] = colors
        pixels[:, :1, :] = colors
        pixels[:, -1:, :] = colors
        pixel_list.append(pixels)
    return pixel_list

def gen_solid_pixels(num_pixels, pixel_width, pixel_height):
    #Generates a list of pixels that alternate between red, green, and blue
    zeros = np.zeros(shape=(pixel_height, pixel_width, 3), dtype = "uint8")
    alpha = np.full(shape = (pixel_height, pixel_width, 1), fill_value = 255, dtype = "uint8")
    pixel_list = []
    for pixel_num in range(num_pixels):
        pixels = np.concatenate((zeros, alpha), axis = 2)
        rgb = int(pixel_num // (num_pixels/3))
        pixels[:, :, rgb] = 255
        pixels[:, :, rgb] = 255
        pixels[:, :, rgb] = 255
        pixels[:, :, rgb] = 255
        pixel_list.append(pixels)
    return pixel_list

if __name__ == "__main__":
    pixel_width = 30
    pixel_height = 30
    pixels = gen_rainbow_pixels(50, pixel_width, pixel_height)
    pixels2 = gen_solid_pixels(50, pixel_width, pixel_height)
    window = WindowWidget(500, 500)

    swap = True
    def button_cb(button):
        global swap
        print(button.text)
        if button.text == "Left is better":
            if swap:
                left_frame.set_pixels(pixels2)
            else:
                left_frame.set_pixels(pixels)
            swap = not swap

    # Buttons
    button_frame = RelativeWidget(parent = window, left = 0, right = 0, top=0, height = 50)
    text = ["Left is better", "Can't tell", "Tie", "Right is better"]
    increment = 1 / len(text)
    for index, t in enumerate(text):
        left = index * increment
        right = 1 - left - increment
        button_holder = RelativeWidget(parent = button_frame, left = left, right = right, top=0, bottom=0)
        button = Button(parent=button_holder, text=t, text_padding = 3)
        button.push_handlers(on_press = button_cb)

    # Pixels
    pixel_frame = RelativeWidget(parent = window, left = 0, right = 0, bottom = 0, top=50)
    left_frame = PixelFrame(parent = pixel_frame, pixel_width = pixel_width, pixel_height = pixel_height, pixel_seq = pixels, left=0, right=0.5, top=0, bottom=0)
    right_frame = PixelFrame(parent = pixel_frame, pixel_width = pixel_width, pixel_height = pixel_height, pixel_seq = pixels, FPS = 25, left=0.5, right=0, top=0, bottom=0)
    

    pyglet.clock.schedule_interval(window.update, 1.0/50)
    pyglet.app.run()

    