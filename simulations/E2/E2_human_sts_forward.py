import numpy as np
import biorbd
import bioviz
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# biorbd_viz = bioviz.Viz("./models/human_sts_arms_nohead.bioMod")

# Load the model
model = biorbd.Model("./models/human_sts_arms_nohead.bioMod")
# model = biorbd_viz

# Execute Bioviz with model for visualization
biorbd_viz = bioviz.Viz(loaded_model=model)

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

Tau = np.ones((ntau,))  # generalized forces

# # Proceed with the forward dynamics
# Qddot = model.ForwardDynamics(Q, Qdot, Tau)
# # Qddot = model.ForwardDynamicsConstraintsDirect(Q, Qdot, Tau)
# print(Qddot.to_array())

# Solve the ODE
# Initial conditions
# Q_0 = np.array([0]);
Q_0 = np.zeros(10)
Qdot_0 = np.ones(10) * 0.5
X_0 = np.hstack((Q_0, Qdot_0))

Tau = np.ones((ntau,))


def fun(_, X):
    q = X[:10]
    qdot = X[10:]
    qddot = model.ForwardDynamics(q, qdot, Tau)
    return np.hstack((qdot, qddot.to_array()))


sol = solve_ivp(fun, [0, 1], X_0, method="RK45", t_eval=np.linspace(0, 1, 300))
print(sol.t)
print(sol.y)

# X = [Q, Qdot]
# Xdot = [Qdot, Qddot]
#
# Xdot = fun(t,X)
# Tau = np.ones((10,)) # generalized forces
# Qddot = model.ForwardDynamics(Q, Qdot, Tau)


# Animate the model
biorbd_viz.load_movement(sol.y[:10])
biorbd_viz.exec()

# Proceed with the inverse dynamics from previously computed accelerations
# Tau_check = model.InverseDynamics(Q, Qdot, Qddot)
# print(Tau_check.to_array())

# Static equilibrium
# Tau_static = model.InverseDynamics(Q, np.zeros(model.nbQ()), np.zeros(model.nbQ()))
# print(Tau_static.to_array())
