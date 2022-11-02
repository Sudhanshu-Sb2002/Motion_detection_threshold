from psychopy import visual, core, event, gui, data, logging
from psychopy.hardware import keyboard
import numpy as np

global WINDOW_DIMENSIONS


class movingShape():
    def __init__(self, shape, velocity):
        self.shape = shape
        self.velocity = velocity

    def detect_collision(self):
        shape = self.shape
        velocity = self.velocity
        if shape.pos[0] + shape.size[0] > WINDOW_DIMENSIONS[0] / 2:
            velocity[0] = -velocity[0]
        if shape.pos[0] - shape.size[0] < -WINDOW_DIMENSIONS[0] / 2:
            velocity[0] = -velocity[0]
        if shape.pos[1] + shape.size[1] > WINDOW_DIMENSIONS[1] / 2:
            velocity[1] = -velocity[1]
        if shape.pos[1] - shape.size[1] < -WINDOW_DIMENSIONS[1] / 2:
            velocity[1] = -velocity[1]
        self.velocity = velocity

    def move(self, dt=1):
        self.shape.pos += self.velocity * dt
        self.detect_collision()


def main():
    global WINDOW_DIMENSIONS
    WINDOW_DIMENSIONS = [1080, 720]
    win = visual.Window(WINDOW_DIMENSIONS, monitor="testMonitor", units="pix", fullscr=False)
    # win.mouseVisible = False
    shape = visual.Rect(win, width=100, height=100, fillColor='red', lineColor='red')
    print(type(shape))
    shape.autoDraw = True
    timer = core.Clock()
    kb = keyboard.Keyboard()
    velocity = np.array([0.05, 0.0])
    while (True):
        shape.pos += velocity

        keylist = kb.getKeys(clear=True)
        if 'q' in keylist:
            break
        if 'w' in keylist:
            # make the shape of rectangle 1.05x
            shape.size = (shape.size[0] * 1.05, shape.size[1] * 1.05)
        elif 's' in keylist:
            # make the shape of rectangle 0.95x
            shape.size = (shape.size[0] * 0.95, shape.size[1] * 0.95)
        if 'a' in keylist:
            velocity *= 0.95
        elif 'd' in keylist:
            velocity *= 1.05
        elif 'v' in keylist:
            velocity = np.array([0.01, 0])
        elif 'z' in keylist:
            velocity *= 10
        elif 'x' in keylist:
            velocity *= 0.1
        if len(keylist) > 0:
            print(velocity)
        # clear keys

        win.flip()


def test_paradigm(window, shape, *, start_velocity=0, block_size=6, **time_intervals):
    # **time_intervals is a dictionary of {"fixation_time","stim_time","inter_trial_time","inter_block_time"}
    fix_time = time_intervals["fixation_time"]
    stim_time = time_intervals["stim_time"]
    inter_trial_time = time_intervals["inter_trial_time"]
    inter_block_time = time_intervals["inter_block_time"]
    # create a clock
    timer = core.Clock()
    # create a keyboard
    kb = keyboard.Keyboard()
    # create a dictionary of velocities and the percentage correct
    velocity_dict = {}


if __name__ == '__main__':
    main()