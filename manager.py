# Manager that starts and signals the end of all of the processes.

from multiprocessing import Process, Queue, Pipe
from time import sleep

from communication import Message
from video_chooser import run_video_chooser
from reward_predictor import run_reward_predictor
from car_racing_agent import run_agent_process

TRAJECTORY_QUEUE_LEN = 5

class CarRacingManager(object):
    def __init__(self):
        self.traj_q = Queue() # Trajectory queue from process 1 to process 2 (unlabeled pairs)
        self.pref_q = Queue() # Labeled trajectory pairs from process 2 to process 3
        self.w_pipes = Pipe() # Pipe to send and receive weights from process 3 to process 1
        self.p_pipes = [] # Pipes owned by processes to communicate with manager
        self.m_pipes = [] # Pipes owned by manager to communicate with processes
        self.processes = []
        self.execute = False
        for _ in range(3):
            ppipe, mpipe = Pipe()
            self.p_pipes.append(ppipe)
            self.m_pipes.append(mpipe)

    def _check_msgs(self):
        # check process 2 (video chooser) pipe
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

    def stop(self):
        # Signal all processes to stop.
        print("Signalling processes to stop...")
        for p in self.m_pipes:
            p.send(Message(sender="mgr", title="stop"))
            
        for p in self.processes:
            p.join()
        print("Done waiting.")
        self.execute = False

    def run_full_program(self):
        # Initialize and start processes
        self.processes.append(Process(target=run_agent_process, args=(self.traj_q,)))
        self.processes.append(Process(target=run_video_chooser, args=(self.traj_q, self.pref_q, self.p_pipes[1])))
        self.processes.append(Process(target=run_reward_predictor, args=(self.pref_q, self.w_pipes[1], self.p_pipes[2])))
        
        for p in self.processes:
            p.start()

        # Manager checks messages and reacts accordingly
        self.execute = True
        while self.execute:
            self._check_msgs()
            sleep(1)
        print("Manager quitting.")
        for p in self.processes:
            p.join()

if __name__ == "__main__":
    mgr = CarRacingManager()
    mgr.run_full_program()
