import numpy as np
import biorbd
import bioviz

# Load the model
model = biorbd.Model("./models/human_sts_arms_nohead.bioMod")

# Execute Bioviz with model for visualization
bioviz.Viz(loaded_model=model).exec()

ntau = model.nbGeneralizedTorque()

# Prepare the model
Q = np.zeros(model.nbQ())  # Set the model position

# Perform the forward kinematics
markers = model.markers(Q)

# Print the results
for marker in markers:
    print(marker.to_array())
    
# Compute accelerations
Qdot = np.zeros(model.nbQ())  # Set the model velocity

Tau = np.zeros((ntau,)) # generalized forces

# Proceed with the forward dynamics
Qddot = model.ForwardDynamics(Q, Qdot, Tau)
#Qddot = model.ForwardDynamicsConstraintsDirect(Q, Qdot, Tau)
print(Qddot.to_array())

# Proceed with the inverse dynamics from previously computed accelerations
Tau_check = model.InverseDynamics(Q, Qdot, Qddot)
print(Tau_check.to_array())

# Static equilibrium
Tau_static = model.InverseDynamics(Q, np.zeros(model.nbQ()), np.zeros(model.nbQ()))
print(Tau_static.to_array())




    
