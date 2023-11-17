import numpy as np
import biorbd
import bioviz

# Load the model
model = biorbd.Model("./models/robotic_manipulator_2d.bioMod")

bioviz.Viz(loaded_model=model).exec()


# Prepare the model
ntau = model.nbGeneralizedTorque()
#Q = np.zeros(model.nbQ())  
Q = np.array([0, 0]) # Set the model position

# Perform the forward kinematics
markers = model.markers(Q)

# Print the marker position for the current configuration
for marker in markers:
    print(marker.to_array())

# Compute accelerations (forward dynamics)
Qdot = np.zeros(model.nbQ())  # Set the model velocity
#Qdot = np.array([0, 0]) # Set the model velocity

Tau = np.zeros((ntau,))
#Tau = np.array([0, 0]) # Set the joint torques for forward dynamics

# Proceed with the forward dynamics
Qddot = model.ForwardDynamics(Q, Qdot, Tau)
print(Qddot.to_array())

# Proceed with the inverse dynamics
# Checking
Tau_check = model.InverseDynamics(Q, Qdot, Qddot)
print(Tau_check.to_array())

# Static equilibrium
Tau_static = model.InverseDynamics(Q, np.zeros(model.nbQ()), np.zeros(model.nbQ()))
print(Tau_static.to_array())
