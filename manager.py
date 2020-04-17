# Manager that starts and signals the end of all of the processes.

from multiprocessing import Process, Queue, Pipe
from time import sleep
import argparse

from communication import Message
from video_chooser import run_video_chooser
from reward_predictor import run_reward_predictor
from car_racing_agent import run_agent_process

TRAJECTORY_QUEUE_LEN = 5

class CarRacingManager(object):
    def __init__(self):
        self.traj_q = Queue() # Trajectory queue from process 1 to process 2 (unlabeled pairs)
        self.pref_q = Queue() # Labeled trajectory pairs from process 2 to process 3
        self.weight_q = Queue(1) # Queue to send and receive weights from process 3 to process 1
        self.p_pipes = [] # Pipes owned by processes to communicate with manager
        self.m_pipes = [] # Pipes owned by manager to communicate with processes
        self.processes = []
        self.execute = False
        for _ in range(3):
            ppipe, mpipe = Pipe()
            self.p_pipes.append(ppipe)
            self.m_pipes.append(mpipe)

    def _check_msgs(self):
        # check process input pipes
        msgs = []
        for pipe in self.m_pipes:
            while pipe.poll():
                msgs.append(pipe.recv())
        self._process_messages(msgs)

    def _process_messages(self, msgs):
        # Handles messages passed to manager from processes
        for msg in msgs:
            if msg.sender == "proc2":
                if msg.title == "close":
                    self.stop()
                elif msg.title == "render":
                    # Forward to proc 1 to render
                    self.m_pipes[0].send(msg)

    def stop(self):
        # Signal all processes to stop.
        print("Waiting for processes to stop...")
        for p in self.m_pipes:
            p.send(Message(sender="mgr", title="stop"))
            
        for p in self.processes:
            p.join()
        print("Done waiting for all processes.")
        self.execute = False

    def run_full_program(self, profile=False, video_FPS=50):
        if profile:
            from os import mkdir
            try:
                mkdir("profile")
            except FileExistsError:
                pass

        # Initialize and start processes
        self.processes.append(Process(target=run_agent_process,
                                      args=(self.traj_q, self.weight_q, self.p_pipes[0]),
                                      kwargs={"profile": profile},
                                      name = "Agent"))
        self.processes.append(Process(target=run_video_chooser,
                                      args=(self.traj_q, self.pref_q, self.p_pipes[1]),
                                      kwargs={"profile": profile, "FPS": video_FPS},
                                      name = "Video Chooser"))
        self.processes.append(Process(target=run_reward_predictor,
                                      args=(self.pref_q, self.weight_q, self.p_pipes[2]),
                                      kwargs={"profile": profile},
                                      name = "Reward Predictor"))
        
        for p in self.processes:
            p.start()

        # Manager checks messages and reacts accordingly
        self.execute = True
        while self.execute:
            self._check_msgs()
            sleep(1)
        print("Manager quitting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--profile", help="Profile the individual processes.", action="store_true")
    args = parser.parse_args()
    mgr = CarRacingManager()
    mgr.run_full_program(profile = args.profile)
