import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- CONFIGURATION: LUBRICANT SELECTOR ---
# dynamic viscosity (mu) in Pascal-seconds (Pa·s)
LUBRICANTS = {
    "Air": 0.000018,    # Very low drag
    "Water": 0.00089,   # Medium drag 
    "Oil": 0.03         # High drag (SAE 30 @ 25°C)
}

# CHOOSE ONE: Uncomment the one you want to use
mu = LUBRICANTS["Air"]
#mu = LUBRICANTS["Air"] + LUBRICANTS["Water"]
#mu = LUBRICANTS["Water"]
#mu = LUBRICANTS["Oil"]

# --- Physical Parameters ---
L, R1, R2 = 0.5, 0.05, 0.052  # 50cm tube, 50mm & 52mm radii
omega_outer = 1.0              # Rotation speed (rad/s)
fps = 30
duration = 30 
total_frames = fps * duration

# --- Torque & Vibration Math ---
# Viscous Torque for concentric cylinders (Taylor-Couette)
viscous_torque = (4 * np.pi * mu * L * (R1**2) * (R2**2) * omega_outer) / (R2**2 - R1**2)

# Generate Time and Noise Data
times = np.linspace(0, duration, total_frames)
# Vibration = Periodic oscillation (bearing waviness) + Random noise (surface roughness)
vibration = (viscous_torque * 0.15 * np.sin(2 * np.pi * 3 * times)) # 3Hz vibration
noise = np.random.normal(0, viscous_torque * 0.05, total_frames)   # 5% random noise
total_torque_data = viscous_torque + vibration + noise

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Cross-Section View
ax1.set_xlim(-0.06, 0.06); ax1.set_ylim(-0.06, 0.06)
ax1.set_aspect('equal')
theta_line = np.linspace(0, 2*np.pi, 200)
ax1.plot(R1*np.cos(theta_line), R1*np.sin(theta_line), 'r-', lw=2, label="Inner (Stationary)")
ax1.plot(R2*np.cos(theta_line), R2*np.sin(theta_line), 'b-', lw=2, label="Outer (Rotating)")
marker, = ax1.plot([], [], 'bo', ms=10) # Shows rotation
torque_txt = ax1.text(-0.055, -0.07, '', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Plot 2: Live Force/Torque Graph
line, = ax2.plot([], [], 'orange', lw=1.5, label="Transferred Torque")
ax2.set_xlim(0, 5)
ax2.set_ylim(min(total_torque_data)*0.8, max(total_torque_data)*1.2)
ax2.set_title(f"Force acting on Inner Ring (using {list(LUBRICANTS.keys())[list(LUBRICANTS.values()).index(mu)]})")
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Torque (Nm)")
ax2.grid(True, alpha=0.3)

def update(i):
    t = times[i]
    # Update rotation marker
    marker.set_data([R2 * np.cos(omega_outer * t)], [R2 * np.sin(omega_outer * t)])
    
    # Update live graph (scrolling window)
    window = 150 
    start = max(0, i - window)
    line.set_data(times[start:i], total_torque_data[start:i])
    if i > window:
        ax2.set_xlim(times[i-window], times[i])
        
    torque_txt.set_text(f"Inst. Torque: {total_torque_data[i]:.6f} Nm")
    return marker, line, torque_txt

ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=True)
plt.legend()
plt.tight_layout()
plt.show()

# To save: ani.save('tube_viscosity_vibration.gif', writer='pillow', fps=fps)
