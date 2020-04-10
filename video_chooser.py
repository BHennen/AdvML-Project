import pyglet

from widgets import WindowWidget, CenteredWidget, RelativeWidget, Button, PixelFrame


# Overall processes:
# A trajectory segment is a sequence of observations and actions, σ = ((o0,a0),(o1,a1),...,(ok−1,ak−1))∈(O×A)k. 
# Write σ1 > σ2 to indicate that the human preferred trajectory segment σ1 to trajectory segment σ2.
#
# These networks are updated by three processes:
# 1. The policy π interacts with the environment to produce a set of trajectories {τ1,...,τi}. The parameters of π 
#    are updated by a traditional reinforcement learning algorithm, in order to maximize the sum of the predicted
#    rewards rt = r(ot, at).
# 2. We select pairs of segments (σ1,σ2) from the trajectories {τ1,...,τi} produced in step 1, and send them to a
#    human for comparison.
# 3. The parameters of the mapping r are optimized via supervised learning to fit the comparisons collected from
#    the human so far.

# This script is for process 2:
# 1) Receive trajectories segment pairs σ1 and σ2 from process 1 into a queue
# 2) Human chooses preference of one trajectory segment over another
# 3) Triple (σ1, σ2, μ) is generated. μ is a distribution over {1,2} indicating which segment the user preferred
#    If the human selects one segment as preferable, then μ puts all of its mass on that choice. If the human 
#    marks the segments as equally preferable, then μ is uniform. Finally, if the human marks the segments as 
#    incomparable, then the comparison is not included in the database.
# 4) Triple is sent to process 3 


class VideoChooser():
    def __init__(self, input_queue, output_queue, FPS=50):
        # Car racing outputs 96x96 pixels at 50 fps. FPS can be changed to make it easier? for human to tell
        self.img_w = 96
        self.img_h = 96
        self.FPS = FPS
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.window = WindowWidget(800, 800, visible=False)
        self.button_text = ["Left is better", "Can't tell", "Tie", "Right is better"]
        self._init_choice_buttons()
        self._init_pixels()

    def _init_choice_buttons(self):
        self.button_frame = RelativeWidget(parent = self.window, left = 0, right = 0, top=0, height = 50)
        increment = 1 / len(self.button_text)
        for index, t in enumerate(self.button_text):
            left = index * increment
            right = 1 - left - increment
            button_holder = RelativeWidget(parent = self.button_frame, left = left, right = right, top=0, bottom=0)
            button = Button(parent=button_holder, text=t, text_padding = 3)
            button.push_handlers(on_press = self._button_cb)

    def _init_pixels(self):
        # Pixels
        self.pixel_frame = RelativeWidget(parent = self.window, left = 0, right = 0, bottom = 0, top=50)
        self.l_pixels = PixelFrame(parent = self.pixel_frame, pixel_width = self.img_w, pixel_height = self.img_h,
                             FPS = self.FPS, left=0, right=0.5, top=0, bottom=0)
        self.r_pixels = PixelFrame(parent = self.pixel_frame, pixel_width = self.img_w, pixel_height = self.img_h,
                             FPS = 25, left=0.5, right=0, top=0, bottom=0)

    def _button_cb(self, button):
        if button.text == "Left is better":
            print("Left chosen")
        elif button.text == "Right is better":
            print("Right chosen")
        elif button.text == "Tie":
            print("Tie chosen")
        elif button.text == "Can't tell":
            print("Can't tell chosen")
            # Do not save triple; discard example.
            pass

    def run(self):
        self.window.window.set_visible()
        pyglet.clock.schedule_interval(self.window.update, 1.0/50) #Update graphics
        pyglet.app.run()

def run_video_chooser():
    chooser = VideoChooser(None, None)
    chooser.run()

if __name__ == "__main__":
    run_video_chooser()