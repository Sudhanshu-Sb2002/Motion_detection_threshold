from psychopy import visual, core

from psychopy.hardware import keyboard
import numpy as np
import os

global WINDOW_DIMENSIONS
global SUBJECT_NAME, SUBJECT_ID
KEYSET = ['f', 'j', 'q', 'space']
KEY_RESPONSES = {'f': -1.0, 'j': 1.0, 'space': 0.0}
INV_KEY_RESPONSES = {-1: "f: left", 1: "j: right", 0: "space: no motion",255: "Wrong Button", 256:" Late/No response"}
INVALID_VALUE = 255
LATE_VALUE=256

OUTPUT_FOLDER = "Outputs"
SUBJECT_ID_FILE = "LatestSubjectID.txt"
ON_CENTER_TRAIN_PATH = "train_on_cen"
ON_CENTER_TEST_PATH = "test_on_cen"
OFF_CENTER_TRAIN_PATH = "train_off_cen"
OFF_CENTER_TEST_PATH = "test_off_cen"


class movingShape():
    def __init__(self, shape, velocity):
        self.shape = shape
        self.velocity = velocity

    def move(self, dt=1):
        self.shape.pos += self.velocity * dt


def block_designer(block_size, stim_types):
    if block_size % len(stim_types) != 0:
        raise ValueError("n_blocks must be a multiple of stim_types")
    block_design = np.zeros(block_size)
    # stim types is -1 0 or 1
    # we choose first stim randomly between -1 and 1, the rest are completely random among  -1 0 and 1
    block_design[0] = np.random.choice([-1, 1])
    block_design[1] = np.random.choice([-1, 0, 1])
    if block_design[1] == 0:
        block_design[2] = np.random.choice([-1, 1])
    else:
        block_design[2] = np.random.choice([-1, 0, 1])

    return np.random.permutation(block_design)


def update_velocity(velocity, actual_responces, expected_responces, ANYLOSS=True):
    no_correct = int(np.sum(actual_responces == expected_responces))
    map_dictionary = {0: 0.003, 1: 0.003, 2: -0.002, 3: -0.002}
    if ANYLOSS == False:
        map_dictionary = {0: 0.005, 1: 0.005, 2: - 0.005, 3: -0.005}
    return round(velocity + map_dictionary[no_correct], 3)


def check_covergence(no_correct, threshold=1, other_threshold=5):
    p = []
    for i in range(4):
        p.append(np.sum(no_correct == i))
        if p[-1] < threshold:
            return False
    if p[0] + p[1] < other_threshold:
        return False
    if p[2] + p[3] < other_threshold:
        return False
    return True

def Instruction_text( fix_time, stim_time, responce_time,off_center=False, training=False):
    pos_text = " top right corner " if off_center else " center "
    train_test_text = " This is the training phase. You will receive feedback after each trial. " if training \
        else " This is the test phase. You will NOT receive feedback after each trial. "

    part_2="2) Then a white square will be shown in the center for " + str(stim_time) + "s. \n"
    if off_center:
        part_2="2) The white score will always be visible, but will start moving once the cross turns yellow " + str(stim_time) + "s. \n"

    return "Instructions: \n\n You will be shown a set of stimuli  and your job is to discriminate " \
    "whether the square is moving left, right or not at all. The stimuli will be shown as follows\n\n" \
    "1) You will first be shown a red cross in the " + pos_text + " of the screen for " + str(fix_time) + "s." \
    "(You HAVE to keep your eyes where the cross was for the ENTIRE DURATION OF THE TRIAL.)\n" +part_2 +\
    "3) After that a yellow circle will appear for " + str(responce_time) +"s. You have to respond if the square was moving left, right or not at all by pressing the corresponding key.\n" \
    "\n\n\n Press 'f' for left, 'j' for right and 'space' for no motion. \n" \
    "(you can locate the f and j keys by the indentation on the corresponding keys) \n\n\n" \
    "Remember to not move your head or shift your gaze from the fixation location. \n\n" + train_test_text + \
    "Press the SPACE KEY if you are ready to start the experiment. \n\n"

def motion_task(window, stim_shape, fix_shape, response_queue,TEXT_STIMULUS, *, speed=0.03, block_size=3, n_blocks=20,
                training=False, fix_time=2, stim_time=1, responce_time=1, interTrialInterval=1, inter_block_time=5,
                feedback_time=5, off_center=False, savepath=None):

    ANYLOSS = False

    # create a clock and keyboard
    timer = core.Clock()
    kb = keyboard.Keyboard()

    # edit the shape
    # fix_shape = visual.Circle(window, radius=10, fillColor='red', lineColor='red')
    fix_shape.opacity = 0
    fix_shape.autoDraw = True
    response_queue.opacity = 0
    response_queue.autoDraw = True
    response_queue.pos = fix_shape.pos

    # stim_shape = movingShape(visual.Rect(window, width=100, height=100, fillColor='red', lineColor='red'), np.array([0.01, 0]))
    stim_shape.shape.opacity = 0
    stim_shape.shape.autoDraw = True

    stim_shape.velocity = np.array([speed, 0])

    # set stimulus velocities, block design and responce arrays
    stim_types = np.array([-1, 0, 1])

    velocities = np.ones((n_blocks, block_size)) * speed

    correct_responses = np.zeros((n_blocks, block_size))
    actual_response = np.zeros((n_blocks, block_size))
    n_correct = np.zeros(n_blocks)

    # clear the keyboard
    kb.getKeys(clear=True)
    # Start the experiment


    TEXT_STIMULUS.autoDraw = True
    TEXT_STIMULUS.opacity = 1

    TEXT_STIMULUS.text = Instruction_text(fix_time, stim_time, responce_time, off_center, training)
    window.flip()
    startkey=kb.waitKeys(keyList=["space",'q'], clear=True)
    if startkey[0].name == 'q':
        #abort the program
        core.quit()
        quit()

    TEXT_STIMULUS.autoDraw = True
    window.flip()
    TEXT_STIMULUS.alignText = 'center'
    for block in range(n_blocks):
        print(speed,end=" ")
        # inter block time
        timer.reset()
        TEXT_STIMULUS.opacity = 1
        while timer.getTime() < inter_block_time:
            TEXT_STIMULUS.text = "Block " + str(block + 1) + " of " + str(n_blocks) + ". \n Block starts in " + str(
                int(inter_block_time - timer.getTime())) + " seconds"
            window.flip()
        TEXT_STIMULUS.opacity = 0
        TEXT_STIMULUS.text = " "
        fix_shape.opacity = 1
        window.flip()

        # decide velocities for positions in the block
        block_design = block_designer(block_size, stim_types)
        correct_responses[block, :] = np.random.permutation(block_design)
        correct_responses[block, :] = np.random.permutation(block_design)
        velocities[block, :] = correct_responses[block, :] * speed

        # Blocks begin
        for trial in range(block_size):
            # inter trial interval
            timer.reset()
            stim_shape.shape.pos = np.array([0, 0])
            if trial>0:
                while timer.getTime() < interTrialInterval:
                    window.flip()

            # fixation
            if off_center:
                stim_shape.shape.opacity = 1
                fix_shape.color='red'
            timer.reset()
            fix_shape.opacity = 1
            while timer.getTime() < fix_time:
                window.flip()

            fix_shape.opacity = 0
            window.flip()
            if off_center:
                fix_shape.opacity = 1.0
                fix_shape.color = 'blue'

            # stimulus
            timer.reset()
            stim_shape.shape.opacity = 1

            stim_shape.velocity = np.array([velocities[block, trial], 0])
            kb.getKeys(clear=True)
            while timer.getTime() < stim_time:
                stim_shape.move()                
                window.flip()
            print(stim_shape.shape.pos/stim_time)
            stim_shape.shape.opacity = 0
            window.flip()

            if off_center:
                fix_shape.opacity = 0
            # queue response
            timer.reset()

            response_queue.opacity = 1
            while timer.getTime() < responce_time:
                window.flip()
            response_queue.opacity = 0
            window.flip()

            # get responces and store them
            keys = kb.getKeys(keyList=KEYSET, clear=True)

            window.flip()
            # check the key responses and store them

            if len(keys) > 0:
                try:
                    if keys[0].name=='q':
                        core.quit()
                        quit()
                    actual_response[block, trial] = KEY_RESPONSES[keys[0].name]

                except ValueError:
                    actual_response[block, trial] = INVALID_VALUE
            else:
                actual_response[block, trial] = LATE_VALUE
            # if training is being performed then  show feedback
            if training:
                TEXT_STIMULUS.opacity = 1
                if actual_response[block, trial] == correct_responses[block, trial]:
                    TEXT_STIMULUS.text = "Correct"
                    TEXT_STIMULUS.color = "green"
                else:
                    TEXT_STIMULUS.text = "Incorrect,\n you pressed"\
                                         + str( INV_KEY_RESPONSES[actual_response[block, trial]]) + \
                                         "correct response was " + INV_KEY_RESPONSES[ correct_responses[block, trial]]
                    TEXT_STIMULUS.color = "red"

                window.flip()
                core.wait(feedback_time)
                TEXT_STIMULUS.text = " "
                TEXT_STIMULUS.color = "white"
                TEXT_STIMULUS.opacity = 0
                window.flip()
            if not ANYLOSS:
                if actual_response[block, trial] != correct_responses[block, trial]:
                    ANYLOSS = True

        n_correct[block] = np.sum(actual_response[block, :] == correct_responses[block, :])
        print(n_correct[block],actual_response[block],correct_responses[block])
        # update the velocity
        if not training:
            speed = update_velocity(speed, correct_responses[block, :], actual_response[block, :], ANYLOSS=ANYLOSS)
        else:
            speed -= 0.02

    converged = check_covergence(n_correct)
    TEXT_STIMULUS.opacity = 1
    TEXT_STIMULUS.text = "Test converged, Results will now be shown"
    if converged == False:
        TEXT_STIMULUS.text = "Failed to converge. Results will now be shown anyway."
    window.flip()
    core.wait(2)
    TEXT_STIMULUS.opacity = 0
    TEXT_STIMULUS.text = " "
    if savepath is not None:
        save_data(savepath, SUBJECT_NAME,SUBJECT_ID, velocities, correct_responses, actual_response, n_correct,
                  converged)
    return velocities, correct_responses, actual_response, n_correct, converged


def take_trial_input():
    print("\nPlease enter subject Name\n")
    subject_name = input()
    ID_FILE_PATH = os.path.join(OUTPUT_FOLDER, SUBJECT_ID_FILE)
    f = open(ID_FILE_PATH, "r", encoding="utf-8")
    last_line = f.readlines()[-1]
    ID = int(last_line.split(" ")[0]) + 1
    print(ID)
    f.close()

    return subject_name, ID


def get_stimuli(win):
    fixation_stimulus = visual.shape.ShapeStim(win, vertices="cross", lineColor='red', fillColor='red', size=50)
    fixation_stimulus.autoDraw = True
    fixation_stimulus.opacity = 0
    stimulus_shape = visual.Rect(win, width=100, height=100, fillColor='white', lineColor='white')
    stimulus_shape.autoDraw = True
    stimulus_shape.opacity = 0
    moving_stimulus = movingShape(stimulus_shape, np.array([0, 0]))
    response_stimlus = visual.Circle(win, radius=5, fillColor='yellow', lineColor='yellow')
    response_stimlus.autoDraw = True
    response_stimlus.opacity = 0
    response_stimlus.pos = fixation_stimulus.pos
    return fixation_stimulus, moving_stimulus, response_stimlus


def save_data(savefolder, subject_name, subject_ID, velocities, correct_responses, actual_responses, trial_performance,
              converged):

    savepath = os.path.join(OUTPUT_FOLDER, savefolder, "subject_" + str(subject_ID))
    print(savepath)
    np.savez(savepath, subject_name=subject_name, subject_ID=subject_ID, velocities=velocities,
             correct_responses=correct_responses, actual_responses=actual_responses,
             trial_performance=trial_performance, converged=converged, allow_pickle=True)


def save_status(ID, Name, success=1):
    if success == -1:
        return
    ID_FILE_PATH = os.path.join(OUTPUT_FOLDER, SUBJECT_ID_FILE)
    f = open(ID_FILE_PATH, "a", encoding="utf-8")
    f.write(str(ID) + " " + Name + " " + str(success) + "\n")
    f.close()


def main():
    global SUBJECT_NAME, SUBJECT_ID
    SUBJECT_NAME, SUBJECT_ID = take_trial_input()

    global WINDOW_DIMENSIONS
    WINDOW_DIMENSIONS = np.array([1366, 768])
    
   
    win = visual.Window(WINDOW_DIMENSIONS, monitor="testMonitor", units="pix", fullscr=True, color='black')

    win.mouseVisible = False
    fixation_stimulus, moving_stimulus, response_stimlus = get_stimuli(win)
    TEXT_STIMULUS = visual.TextStim(win, text=" ", pos=(0, 0), wrapWidth=700, alignText='left')
    '''# training
    motion_task(win, moving_stimulus,fixation_stimulus,response_stimlus,TEXT_STIMULUS,speed=0.08, n_blocks=4,training=True,
                 savepath=ON_CENTER_TRAIN_PATH)'''

    # test
    motion_task(win, moving_stimulus,fixation_stimulus,response_stimlus,TEXT_STIMULUS,speed=0.04, n_blocks=16, training=False,
                 savepath=ON_CENTER_TEST_PATH)

    fixation_stimulus.pos = np.array([WINDOW_DIMENSIONS / 2 - 50])
    response_stimlus.pos = fixation_stimulus.pos

    # off center training
    motion_task(win, moving_stimulus, fixation_stimulus, response_stimlus,TEXT_STIMULUS, speed=0.8, n_blocks=2,training=True,
                off_center=True,savepath=OFF_CENTER_TRAIN_PATH)

    # off center test
    motion_task(win, moving_stimulus,fixation_stimulus,response_stimlus,TEXT_STIMULUS, speed=0.8, n_blocks=16,training=False,
                 off_center=True, savepath=OFF_CENTER_TEST_PATH)

    win.close()
    status = int(input("Enter 1 if successful, 0 if not,2 if this is a test response, -1 to not save: "))
    save_status(SUBJECT_ID,SUBJECT_NAME, status)


if __name__ == '__main__':
    main()




