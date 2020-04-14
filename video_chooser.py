
from widgets import *


import pyglet
from queue import Full, Empty
from time import sleep
from multiprocessing import Process, Queue, current_process, freeze_support

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
    '''

    '''
    def __init__(self, input_queue, output_queue, on_close_cb, FPS=50):
        '''
        input_queue and output_queue are Queue objects from the multiprocessing package
        '''
        # Car racing outputs 96x96 pixels at 50 fps. FPS can be changed to make it easier? for human to tell
        self.img_w = 96
        self.img_h = 96
        self.FPS = FPS
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.trajectory_1 = self.trajectory_2 = None
        self.window = WindowWidget(800, 800, visible=False)
        self.window.window.push_handlers(on_close = on_close_cb)
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
                             FPS = self.FPS, left=0.5, right=0, top=0, bottom=0)

    def _enable_buttons(self, dt):
        self.button_enabled = True

    def _button_cb(self, button):
        if not self.button_enabled:
            print("Watch video.")
            return
        else:
            self.button_enabled = False

        if button.text == "Left is better":
            print("Left chosen")
            preference = [1, 0]
            self._save_triple(self.trajectory_1, self.trajectory_2, preference)
        elif button.text == "Right is better":
            print("Right chosen")
            preference = [0, 1]
            self._save_triple(self.trajectory_1, self.trajectory_2, preference)
        elif button.text == "Tie":
            print("Tie chosen")
            preference = [0.5, 0.5]
            self._save_triple(self.trajectory_1, self.trajectory_2, preference)
        elif button.text == "Can't tell":
            print("Can't tell chosen")
            # Do not save triple; discard example.
            pass
        
        # Allow user to choose new pair of trajectories.
        self._get_new_pair()

    def _get_new_pair(self, dt=None):
        # Fetch a new pair of observations from the input queue
        if not dt:
            self.l_pixels.wait()
            self.r_pixels.wait()
            self.window.window.flip()
        try:
            self.trajectory_1 = self.trajectory_2 = None
            trajectory_1, trajectory_2 = self.input_queue.get_nowait()
        except Empty:
            # Loop back around to get more after a short amount of time to respond to window events
            pyglet.clock.schedule_once(self._get_new_pair, 1)
            return
        obs_1 = [obs for obs, action in trajectory_1]
        obs_2 = [obs for obs, action in trajectory_2]
        self.l_pixels.set_pixels(obs_1)
        self.r_pixels.set_pixels(obs_2)
        self.trajectory_1, self.trajectory_2 = trajectory_1, trajectory_2
        pyglet.clock.schedule_once(self._enable_buttons, 0.5)

    def _save_triple(self, trajectory_1, trajectory_2, preference):
        try:
            self.output_queue.put_nowait([trajectory_1, trajectory_2, preference])
        except Full:
            # If the output queue is full remove an item and save it again.
            print("process 3 busy, queue full; removing item from prefence output queue")
            self.output_queue.get_nowait()
            self._save_triple(trajectory_1, trajectory_2, preference)

    def run(self):
        self.window.window.set_visible()
        self._get_new_pair()
        pyglet.clock.schedule_interval(self.window.update, 1.0/self.FPS) #Update graphics
        pyglet.app.run()

def run_video_chooser(input_queue, output_queue, on_close_cb):
    chooser = VideoChooser(input_queue, output_queue, on_close_cb)
    chooser.run()

def window_closed_test():
    print("window closed")

def test():
    # Create queues
    trajectory_input_queue = Queue()
    preference_output_queue = Queue()

    num_pixels=30
    pixel_width=96
    pixel_height=96
    rainbow_obs = gen_rainbow_pixels(num_pixels, pixel_width, pixel_height)
    solid_obs = gen_solid_pixels(num_pixels, pixel_width, pixel_height)

    traj_1 = list(zip(rainbow_obs, [0] * len(rainbow_obs)))
    traj_2 = list(zip(solid_obs, [1] * len(solid_obs)))

    # Generate some trajectories
    for i in range(3):
        if i % 2 == 0:
            trajectory_input_queue.put([traj_1, traj_2])
        else:
            trajectory_input_queue.put([traj_2, traj_1])
    
    p = Process(target=run_video_chooser, args=(trajectory_input_queue, preference_output_queue, window_closed_test))
    p.start()

    #Wait for user to choose
    # NOTE: If can't tell is chosen, this process will block.
    for i in range(3):
        trajectory_1, trajectory_2, preference = preference_output_queue.get()
        print(preference)
    
    sleep(5)

    # Generate some more trajectories
    for i in range(3):
        if i % 2 == 0:
            trajectory_input_queue.put([traj_1, traj_2])
        else:
            trajectory_input_queue.put([traj_2, traj_1])

    #Wait for user to choose
    for i in range(3):
        trajectory_1, trajectory_2, preference = preference_output_queue.get()
        print(preference)


if __name__ == "__main__":
    test()