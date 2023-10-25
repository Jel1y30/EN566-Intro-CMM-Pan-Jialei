import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.8
l = 9.8
gamma = 0.25
omegaD = 0.666
delta_theta_in = 0.001

# Time evolution settings
dt = 0.01
t_max = 100
t = np.arange(0, t_max, dt)

def euler_cromer_nonlinear(initial_theta, omegaD, alphaD):
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)
    theta[0] = initial_theta
    for i in range(1, t.size):
        omega[i] = omega[i-1] + (-g/l*np.sin(theta[i-1]) - 2*gamma*omega[i-1] + alphaD*np.sin(omegaD*t[i-1])) * dt
        theta[i] = theta[i-1] + omega[i] * dt
    return theta

fig, ax = plt.subplots(figsize=(10, 6))

for alphaD in [0.2, 0.5, 1.2]:
    # Two trajectories
    theta1 = euler_cromer_nonlinear(0, omegaD, alphaD)
    theta2 = euler_cromer_nonlinear(delta_theta_in, omegaD, alphaD)

    # Compute |Δθ(t)|
    delta_theta = np.abs(theta1 - theta2)

    # Plot |Δθ(t)|
    ax.plot(t, delta_theta, label=f'αD = {alphaD} rad/s^2')

ax.set_title('|Δθ(t)| vs. time')
ax.set_xlabel('Time (s)')
ax.set_ylabel('|Δθ(t)| (radians)')
ax.legend()

plt.show()
