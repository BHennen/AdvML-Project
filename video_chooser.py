
from widgets import WindowWidget, RelativeWidget, PixelFrame, Button
from communication import Message

import pyglet

import os
from queue import Full, Empty
from time import sleep
from multiprocessing import Process, Queue, current_process

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
    def __init__(self, traj_q, pref_q, mgr_conn, FPS=50):
        '''
        traj_q and pref_q are Queue objects from the multiprocessing package
        '''
        # Car racing outputs 96x96 pixels at 50 fps. FPS can be changed to make it easier? for human to tell
        self.img_w = 96
        self.img_h = 96
        self.FPS = FPS
        self.traj_q = traj_q
        self.pref_q = pref_q
        self.mgr_conn = mgr_conn
        self.trajectory_1 = self.trajectory_2 = None
        self.window = WindowWidget(800, 800, visible=False)
        self.window.window.push_handlers(on_close = self._on_close)
        self.button_text = ["Left is better", "Can't tell", "Tie", "Right is better", "Render"]
        self._init_choice_buttons()
        self._init_pixels()

    def _on_close(self):
        # Window requested to be closed
        if self.mgr_conn is None:
            pass # No manager connection so close the window.
        else:
            self.mgr_conn.send(Message(sender="proc2", title="close"))
            return True

    def _check_msgs(self, dt):
        # Checks manager messages 
        if self.mgr_conn is None:
            return
        
        msgs = []
        while self.mgr_conn.poll():
            msgs.append(self.mgr_conn.recv())
        
        # see if we need to close the window
        for msg in msgs:
            if msg.sender == "mgr":
                if msg.title == "stop":
                    self._stop()
    
    def _stop(self):
        # Remove handlers
        for holder in self.button_frame.children:
            for button in holder.children:
                button.pop_handlers()
        self.window.close() # Signal window to close
        self.window.window.close() # Actually close window
        # close pipe
        self.mgr_conn.close()
        pyglet.app.exit()
        print(f"Quitting {current_process().name} process")
        os._exit(0)

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
        if button.text == "Render":
            self.mgr_conn.send(Message(sender="proc2", title="render"))
            return # don't get new pair
        
        if self.trajectory_1 is None:
            print("Wait for clips.")
            return

        if not self.button_enabled:
            return
        else:
            self.button_enabled = False

        if button.text == "Left is better":
            preference = [1, 0]
            self._save_triple(self.trajectory_1, self.trajectory_2, preference)
        elif button.text == "Right is better":
            preference = [0, 1]
            self._save_triple(self.trajectory_1, self.trajectory_2, preference)
        elif button.text == "Tie":
            preference = [0.5, 0.5]
            self._save_triple(self.trajectory_1, self.trajectory_2, preference)
        elif button.text == "Can't tell":
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
        
        self.trajectory_1 = self.trajectory_2 = None
        try:
            trajectory_1, trajectory_2 = self.traj_q.get_nowait()
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
            self.pref_q.put_nowait([trajectory_1, trajectory_2, preference])
        except Full:
            # If the output queue is full remove an item and save it again.
            print("process 3 busy, queue full; removing item from prefence output queue")
            self.pref_q.get_nowait()
            self._save_triple(trajectory_1, trajectory_2, preference)

    def _run(self):
        self.window.window.set_visible()
        self._get_new_pair()
        pyglet.clock.schedule_interval(self._check_msgs, 1)
        pyglet.clock.schedule_interval(self.window.update, 1.0/self.FPS) #Update graphics
        pyglet.app.run()

def run_video_chooser(traj_q, pref_q, mgr_conn):
    chooser = VideoChooser(traj_q, pref_q, mgr_conn)
    chooser._run()

def test():
    from widgets import gen_rainbow_pixels, gen_solid_pixels
    # Create queues
    traj_q = Queue()
    pref_q = Queue()

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
            traj_q.put([traj_1, traj_2])
        else:
            traj_q.put([traj_2, traj_1])
    
    p = Process(target=run_video_chooser, args=(traj_q, pref_q, None))
    p.start()

    #Wait for user to choose
    # NOTE: If can't tell is chosen, this process will block.
    for i in range(3):
        trajectory_1, trajectory_2, preference = pref_q.get()
        print(preference)
    
    sleep(5)

    # Generate some more trajectories
    for i in range(3):
        if i % 2 == 0:
            traj_q.put([traj_1, traj_2])
        else:
            traj_q.put([traj_2, traj_1])

    #Wait for user to choose
    for i in range(3):
        trajectory_1, trajectory_2, preference = pref_q.get()
        print(preference)


if __name__ == "__main__":
    test()