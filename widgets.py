import pyglet
import numpy as np
from colorsys import hsv_to_rgb

from image_manager import ImageManager

# Create viewer class GUI that implements:
# TODO: buttons (to prefer left sequence, right sequence, neither, or tie)

class Widget(object):
    def __init__(self, parent, x, y, width, height):
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

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        child.resize()

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
        def scale(percentage, value):
            if percentage is None:
                return None
            elif abs(percentage) > 1:
                return percentage
            else:
                return percentage * value
        
        left = scale(self._left, self.parent.width)
        right = scale(self._right, self.parent.width)
        top = scale(self._top, self.parent.height)
        bottom = scale(self._bottom, self.parent.height)
        width = scale(self._set_width, self.parent.width)
        height = scale(self._set_height, self.parent.height)

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
        window = pyglet.window.Window(window_width, window_height, resizable=True)
        window.push_handlers(on_resize = self._on_resize, on_draw = self.render)

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


class PixelFrame(RelativeWidget):
    def __init__(self, parent, pixel_width, pixel_height, pixel_seq, FPS=50, left=None, right=None, top=None, bottom=None, width=None, height=None):
        super().__init__(parent, left=left, right=right, top=top, bottom=bottom, width=width, height=height)
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width
        self.mgr = ImageManager(img_height = pixel_height, img_width = pixel_width, max_texture_size=1024)
        self.set_pixels(pixel_seq)
        self.num_imgs = len(pixel_seq)
        self.img_index = 0
        self.switch_time = 1/FPS 
        self.countdown = self.switch_time

    def set_pixels(self, pixel_seq):
        print("Loading pixels...", end="", flush=True)
        for index, pixels in enumerate(pixel_seq):
            self.mgr.update_image(pixels, index)
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

if __name__ == "__main__":
    # Generate frames and pixels before window
    pixel_width = 30
    pixel_height = 30
    pixels = gen_rainbow_pixels(50, pixel_width, pixel_height)
    pixel_frames = RelativeWidget(parent = None, left = 0, right = 0, bottom = 0, top=30)
    left_frame = PixelFrame(parent = pixel_frames, pixel_width = pixel_width, pixel_height = pixel_height, pixel_seq = pixels, left=0, right=0.5, top=0, bottom=0)
    right_frame = PixelFrame(parent = pixel_frames, pixel_width = pixel_width, pixel_height = pixel_height, pixel_seq = pixels, FPS = 25, left=0.5, right=0, top=0, bottom=0)

    window = WindowWidget(500, 500)
    window.add_child(pixel_frames)

    pyglet.clock.schedule_interval(window.update, 1.0/50)
    pyglet.app.run()

    