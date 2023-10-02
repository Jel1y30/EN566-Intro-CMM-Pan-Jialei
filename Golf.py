import numpy as np
import matplotlib.pyplot as plt

def trajectory(theta, case):
    u = 70  # m/s
    g = 9.81  # m/s^2
    rho = 1.29  # kg/m^3
    A = 0.0014  # m^2
    m = 0.046  # kg
    S0w_by_m = 0.25  # s^-1
    dt = 0.01  # time step
    x, y = [0], [0]
    vx = u * np.cos(np.radians(theta))
    vy = u * np.sin(np.radians(theta))

    while y[-1] >= 0:
        if case == "ideal":
            ax = 0
            ay = -g
        else:
            v = np.sqrt(vx**2 + vy**2)
            if case == "smooth":
                C = 0.5
            elif case == "dimpled":
                C = 0.5 if v < 14 else 7.0/v
            else:  # dimpled with spin
                C = 0.5 if v < 14 else 7.0/v
                FMagnus = S0w_by_m * m * v
                ayMagnus = FMagnus * np.cos(np.radians(theta)) / m
                axMagnus = -FMagnus * np.sin(np.radians(theta)) / m
            Fdrag = -C * rho * A * v**2
            axDrag = Fdrag * np.cos(np.radians(theta)) / m
            ayDrag = Fdrag * np.sin(np.radians(theta)) / m
            ax = axDrag + (axMagnus if case == "dimpled_spin" else 0)
            ay = -g + ayDrag + (ayMagnus if case == "dimpled_spin" else 0)
        vx += ax * dt
        vy += ay * dt
        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)
    return x, y

angles = [45, 30, 15, 9]
cases = ["ideal", "smooth", "dimpled", "dimpled_spin"]

plt.figure(figsize=(15, 8))
for theta in angles:
    for case in cases:
        x, y = trajectory(theta, case)
        plt.plot(x, y, label=f"{theta}°, {case}")
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.title('Trajectories of a Golf Ball')
plt.legend()
plt.grid(True)
plt.show()
