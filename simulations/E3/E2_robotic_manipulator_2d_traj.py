import numpy as np
import biorbd
import bioviz

biorbd_viz = bioviz.Viz("./models/robotic_manipulator_2d.bioMod")

manually_animate = False

# Load the model
model = biorbd.Model("/models/robotic_manipulator_2d.bioMod")

# Generate pick-and-place motion
n_frames = 200
qinit = np.array([-np.pi/3, 2*np.pi/3])
qfinal = np.array([0, np.pi/3])
q_all = np.linspace(qinit, qfinal, n_frames).T;

# Animate the model
biorbd_viz.load_movement(q_all)
biorbd_viz.exec()

