import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import pandas

joints = pandas.read_csv("../data/out.csv")
grouped = joints.values.reshape((len(joints), 32, 3))

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

graph, = ax.plot(grouped[0, :, 0].flatten(), grouped[0, :, 2].flatten(), (-1)*grouped[0, :, 1].flatten(), linestyle="", marker="o")

def update_graph(num):
    graph.set_data (grouped[num, :, 0].flatten(), grouped[num, :, 2].flatten())
    graph.set_3d_properties((-1)*grouped[num, :, 1].flatten())
    title.set_text('3D Test, time={}'.format(num))
    return title, graph,


# Setting the axes properties
ax.set_xlim3d([-0.75, 0.75])
ax.set_xlabel('X')

ax.set_ylim3d([1.0, 2.5])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')

title = ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_graph, len(joints),
                                   interval=50, blit=False)

plt.show()
