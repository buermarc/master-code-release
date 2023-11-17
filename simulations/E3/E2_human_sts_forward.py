import numpy as np
import biorbd
import bioviz
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#biorbd_viz = bioviz.Viz("./models/human_sts_arms_nohead.bioMod")
biorbd_viz = bioviz.Viz("./models/human_sts_arms.bioMod")

# Load the model
#model = biorbd.Model("./models/human_sts_arms_nohead.bioMod")
model = biorbd.Model("./models/human_sts_arms.bioMod")

# Execute Bioviz with model for visualization
#bioviz.Viz(loaded_model=model).exec()

ntau = model.nbGeneralizedTorque()

# Prepare the model
Q = np.zeros(model.nbQ())  # Set the model position

# Perform the forward kinematics
markers = model.markers(Q)

# Print the results
# for marker in markers:
#     breakpoint()
#     type(marker.to_array())
#     print(marker.to_array())

# Compute accelerations
Qdot = np.zeros(model.nbQ())  # Set the model velocity

# Proceed with the forward dynamics
#Qddot = model.ForwardDynamics(Q, Qdot, Tau)
#Qddot = model.ForwardDynamicsConstraintsDirect(Q, Qdot, Tau)
#print(Qddot.to_array())

# Solve the ODE
# Initial conditions
#Q_0 = np.array([0]);
Q_0 = np.zeros(model.nbQ())
Qdot_0 = np.ones(model.nbQ())*0.2
Qdot_0 = np.zeros(model.nbQ())
Qdot_0[6] = -0.3
Qdot_0[7] = 0.8
Qdot_0[9] = 0.3
Qdot_0[10] = -0.8
X_0 = np.hstack((Q_0, Qdot_0))

Tau = np.ones((ntau,))*0.1
Tau = np.zeros((ntau,))

def fun(t, X):
    Q = X[:model.nbQ()]
    Qdot = X[model.nbQ():]
    # Recalculate the tau which would lead to static condition
    Qddot = model.ForwardDynamics(Q, Qdot, Tau)

    return np.hstack((Qdot, Qddot.to_array()))

end = 1
samples = 30 * end
sol = solve_ivp(fun, [0, end], X_0, method='RK45', t_eval=np.linspace(0, end, samples))
print(sol.t)
print(sol.y)

#X = [Q, Qdot]
#Xdot = [Qdot, Qddot]

#Xdot = fun(t,X)
#Tau = np.ones((10,)) # generalized forces
#Qddot = model.ForwardDynamics(Q, Qdot, Tau)

# for frame in sol.y[:model.nbQ()].T:
#     for marker in model.markers(frame):
#         print(marker.to_array())

# Animate the model
biorbd_viz.load_movement(sol.y[:model.nbQ()])
biorbd_viz.exec()


# Proceed with the inverse dynamics from previously computed accelerations
#Tau_check = model.InverseDynamics(Q, Qdot, Qddot)
#print(Tau_check.to_array())

# Static equilibrium
#Tau_static = model.InverseDynamics(Q, np.zeros(model.nbQ()), np.zeros(model.nbQ()))
#print(Tau_static.to_array())
