from pyglet.gl import * #pylint: disable=unused-wildcard-import
import numpy as np
from ctypes import byref
from colorsys import hsv_to_rgb

class ImageManager():
    '''
    Manages rectangular images and stores them in larger textures which can be accessed sequentially.

    Assumes textures share the same dimensions.

    img_height: The height of images
    img_width: The width of images
    max_texture_size: The internal maximum size of the texture. Defaults to GL_MAX_TEXTURE_SIZE.
    '''
    # TODO: manage overwriting of images (faster to modify existing texture than to make new ones)
    def __init__(self, img_height, img_width, max_texture_size=None):
        self.max_texture_size = max_texture_size or GL_MAX_TEXTURE_SIZE
        self.img_height = img_height
        self.img_width = img_width
        self.texture_ids = []
        self.num_imgs = 0
        self.imgs_per_row = self.max_texture_size // img_width
        self.imgs_per_col = self.max_texture_size // img_height
        self.imgs_per_tex = self.imgs_per_row * self.imgs_per_col

    def append_image(self, pixel_data):
        '''
        Adds a new image to the image manager.

        Assumes data has the shape (img_height, img_width, 4), where 4 is RGBA pixel values.

        Also assumes the data should be output from left to right, top to bottom
        '''

        tex_id = None
        glEnable(GL_TEXTURE_2D)

        # Create new texture (if needed)
        if self.num_imgs / self.imgs_per_tex >= len(self.texture_ids):
            tex_id = self._create_empty_texture()

        # Assign last used tex_id and bind it.
        if not tex_id:
            tex_id = self.texture_ids[-1]
            glBindTexture(GL_TEXTURE_2D, tex_id.value)

        # Format data
        pixels = np.flip(pixel_data, axis=0).flatten() # Flip so the image is right side up
        tex_data = (GLubyte * pixels.size)( *pixels.astype('uint8'))
        

        # Get x and y offset and save data to correct spot in texture
        x_offset, y_offset, _ = self._get_offset(self.num_imgs)
        # Add image to the next open spot
        glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, self.img_width)
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0)
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 0)
        glTexSubImage2D(GL_TEXTURE_2D,      # target,            
                        0,                  # level,
                        x_offset,           # xoffset,
                        y_offset,           # yoffset,
                        self.img_width,     # width,        
                        self.img_height,    # height,            
                        GL_RGBA,            # format,        
                        GL_UNSIGNED_BYTE,   # type,                
                        tex_data)           # pixels);        
        glPopClientAttrib()
        glDisable(GL_TEXTURE_2D)
        self.num_imgs += 1

    def draw_image(self, index, x, y, width=None, height=None):
        '''
        Draws an image at the given x and y coordinate. Default width and height is the image size specified in the constructor.
        '''
        if index >= self.num_imgs:
            raise IndexError(f"draw_image: Index out of bounds.")

        # Draws the image specified by index to the screen
        x_t_offset, y_t_offset, texture_ind = self._get_offset(index)
        tex_id = self.texture_ids[texture_ind]
        # Texture coordinate positions
        x_t1 = (x_t_offset) / self.max_texture_size
        y_t1 = (y_t_offset) / self.max_texture_size
        x_t2 = (x_t_offset + self.img_width) / self.max_texture_size
        y_t2 = (y_t_offset + self.img_height) / self.max_texture_size
        # image (vector) coordinate positions
        x_v1 = x
        y_v1 = y
        x_v2 = x_v1 + (width is None and self.img_width or width)
        y_v2 = y_v1 + (height is None and self.img_height or height)

        array = (GLfloat * 48)(
            # 1 - repeated for degenerate triangle
            x_t1, y_t1, 0, 1.,
            x_v1, y_v1, 0, 1.,
            x_t1, y_t1, 0, 1.,
            x_v1, y_v1, 0, 1.,
            # 2
            x_t2, y_t1, 0, 1.,
            x_v2, y_v1, 0, 1.,
            # 3
            x_t1, y_t2, 0, 1.,
            x_v1, y_v2, 0, 1.,
            # 4 - repeated for degenerate triangle
            x_t2, y_t2, 0, 1.,
            x_v2, y_v2, 0, 1.,
            x_t2, y_t2, 0, 1.,
            x_v2, y_v2, 0, 1.)
        
        glPushAttrib(GL_ENABLE_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glInterleavedArrays(GL_T4F_V4F, 0, array)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 6)
        glPopClientAttrib()
        glPopAttrib()
        glDisable(GL_TEXTURE_2D)

    def _get_texel_coord(self, pixel_ind):
        # Note: Not needed for some reason
        return (pixel_ind + 0.5) / self.max_texture_size

    def _get_offset(self, index):
        # Return the (x, y, z) pixel offset of the given image index,
        # where x is the x_offset, y is the y_offset, and z is the 
        # texture the image is located in.
        x = (index % self.imgs_per_row)
        y = (index // self.imgs_per_row) % self.imgs_per_col
        z = index // self.imgs_per_tex
        return (x * self.img_width, y * self.img_height, z)

    def _create_empty_texture(self):
        # Create an empty texture and returns the texture ID
        tex_id = GLuint()
        target = GL_TEXTURE_2D
        glGenTextures(1, byref(tex_id))
        glBindTexture(target, tex_id.value)
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        blank = (GLubyte * (self.max_texture_size * self.max_texture_size * 4))()
        glTexImage2D(target, 0,
                     GL_RGBA,
                     self.max_texture_size, self.max_texture_size,
                     0,
                     GL_RGBA, GL_UNSIGNED_BYTE,
                     blank)
        self.texture_ids.append(tex_id)
        return tex_id


class ImageManagerTester():
    def __init__(self, draw_height, draw_width, pixel_height, pixel_width, mode):
        self.draw_height = draw_height
        self.draw_width = draw_width
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width
        self.mode = mode
        self.num_modes = 2

        if self.mode % self.num_modes == 1:
            self.switch_time = 1/50 #fps
            self.countdown = self.switch_time
        else:
            self.switch_time = 2
            self.countdown = self.switch_time
            
        self.img_index = 0
        self.mgr = ImageManager(img_height = pixel_height, img_width = pixel_width, max_texture_size = 1024)
        self.num_imgs = min(self.mgr.imgs_per_tex + 1, 100) #cap number of generated pixels
        self._init_pixels()

    def update(self, dt):
        # Cycle through the images in the image manager
        self.countdown -= dt
        # Skip some images if too much time has passed
        while self.countdown < 0:
            self.countdown += self.switch_time
            self.img_index = (self.img_index+1) % self.num_imgs

    def render(self):
        self.mgr.draw_image(self.img_index, 0, 0, self.draw_height, self.draw_width)

    def _init_pixels(self):        
        zeros = np.zeros(shape=(self.pixel_height, self.pixel_width, 3), dtype = "uint8")
        alpha = np.full(shape = (self.pixel_height, self.pixel_width, 1), fill_value = 255, dtype = "uint8")
        # Create enough images to fill a texture + 1
        # images will be solid borders alternating between red blue and green
        print(f"Init {self.num_imgs} pixel arrays")
        for pixel_num in range(self.num_imgs):
            if pixel_num >= 100:
                break
            pixels = np.concatenate((zeros, alpha), axis = 2)
            if self.mode % self.num_modes == 0:
                rgb = pixel_num % 3
                pixels[:1, :, rgb] = 255
                pixels[-1:, :,rgb] = 255
                pixels[:, :1, rgb] = 255
                pixels[:, -1:, rgb] = 255
            elif self.mode % self.num_modes == 1:
                hue = pixel_num / self.num_imgs
                rgb = hsv_to_rgb(h=hue, s=1, v=1)
                colors = [col*255 for col in rgb] + [255]
                pixels[:1, :, :] = colors
                pixels[-1:, :, :] = colors
                pixels[:, :1, :] = colors
                pixels[:, -1:, :] = colors
            self.mgr.append_image(pixels)

PIXEL_WIDTH = 96
PIXEL_HEIGHT = 96
VIEWER_WIDTH = 300
VIEWER_HEIGHT = 300
FPS = 50

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
        for obj in self.objs:
            obj.render()

    def add_obj(self, obj):
        self.objs.append(obj)
    
    def update(self, dt):
        for obj in self.objs:
            obj.update(dt)

if __name__ == "__main__":

    VIEWERS = []

    def update(dt):
        for viewer in VIEWERS:
            viewer.update(dt)

    mgr_test_1 = ImageManagerTester(draw_height = VIEWER_HEIGHT, draw_width = VIEWER_WIDTH, pixel_height = PIXEL_HEIGHT, pixel_width = PIXEL_WIDTH, mode = 0)
    win1 = Viewer()
    win1.add_obj(mgr_test_1)

    mgr_test_2 = ImageManagerTester(draw_height = VIEWER_HEIGHT, draw_width = VIEWER_WIDTH, pixel_height = PIXEL_HEIGHT, pixel_width = PIXEL_WIDTH, mode = 1)
    win2 = Viewer()
    win2.add_obj(mgr_test_2)

    VIEWERS.extend([win1, win2])

    pyglet.clock.schedule_interval(update, 1.0/FPS)
    pyglet.app.run()