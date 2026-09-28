import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import center_of_mass

# --- Simulation Parameters ---
GRID_SIZE = 128
DT = 0.02         # Even smaller DT for more stability
DX = 1.0
C_WAVE = 1.0

# --- Particle Parameters ---
PARTICLE_AMPLITUDE = 1.5 # Amplitude should relate to stable points of V(phi)
PARTICLE_RADIUS = 10 # Make it a bit wider
PARTICLE_INITIAL_POS_XY = (GRID_SIZE * 0.25, GRID_SIZE * 0.5) # Start further left
PARTICLE_VELOCITY_XY = (0.25, 0.0) # Slower initial velocity

# --- PDE Parameters ---
# For potential V(phi) = -a/2 * phi^2 + b/4 * phi^4 (Mexican Hat)
# Stable non-zero states are at phi = +/- sqrt(a/b)
# Force F = -dV/dphi = a*phi - b*phi^3
# We set PARTICLE_AMPLITUDE near sqrt(A_COEFF_POTENTIAL / B_COEFF_POTENTIAL)
A_COEFF_POTENTIAL = 0.1  # 'a' in force term (positive)
B_COEFF_POTENTIAL = 0.1  # 'b' in force term (positive)
# With these, stable states are near +/- sqrt(0.1/0.1) = +/- 1.0. So PARTICLE_AMPLITUDE 1.5 is okay.

class SubstrateParticleSimulator:
    def __init__(self, grid_size, dt, dx, c_wave, a_potential, b_potential):
        self.grid_size = grid_size
        self.dt = dt
        self.dx = dx
        self.c_wave = c_wave
        self.a_potential = a_potential # Renamed for clarity
        self.b_potential = b_potential # Renamed for clarity

        self.phi = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.phi_prev = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.time = 0.0

    def _initialize_particle(self, center_x, center_y, amplitude, radius):
        y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
        dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
        width_param = radius 
        profile = amplitude / np.cosh(np.sqrt(dist_sq) / width_param)
        self.phi += profile

    def _apply_initial_velocity(self, velocity_x, velocity_y):
        phi_grad_x = (np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / (2 * self.dx)
        phi_grad_y = (np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / (2 * self.dx)
        phi_t_initial = -velocity_x * phi_grad_x - velocity_y * phi_grad_y
        self.phi_prev = self.phi - phi_t_initial * self.dt
        if velocity_x == 0 and velocity_y == 0:
            self.phi_prev = self.phi.copy()

    def _laplacian(self, field):
        lap_x = (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / self.dx**2
        lap_y = (np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / self.dx**2
        return lap_x + lap_y

    def step(self):
        lap_phi = self._laplacian(self.phi)

        force_substrate_waves = self.c_wave**2 * lap_phi
        
        # Force from potential: F = a*phi - b*phi^3
        force_from_potential = self.a_potential * self.phi - self.b_potential * self.phi**3

        phi_new = 2 * self.phi - self.phi_prev + \
                  self.dt**2 * (force_substrate_waves + force_from_potential)

        self.phi_prev = self.phi.copy()
        self.phi = phi_new
        self.time += self.dt

    def setup_simulation(self):
        self._initialize_particle(PARTICLE_INITIAL_POS_XY[0], PARTICLE_INITIAL_POS_XY[1],
                                  PARTICLE_AMPLITUDE, PARTICLE_RADIUS)
        if PARTICLE_VELOCITY_XY[0] != 0 or PARTICLE_VELOCITY_XY[1] != 0:
            self._apply_initial_velocity(PARTICLE_VELOCITY_XY[0], PARTICLE_VELOCITY_XY[1])
        else:
            self.phi_prev = self.phi.copy()

# --- Visualization ---
simulator = SubstrateParticleSimulator(GRID_SIZE, DT, DX, C_WAVE, A_COEFF_POTENTIAL, B_COEFF_POTENTIAL)
simulator.setup_simulation()

fig, ax = plt.subplots(figsize=(10,8))
ax.set_title("Particle-Field Sailing on Substrate (Φ)")

# Determine vmin/vmax based on potential stable points and particle amplitude
stable_phi_abs = np.sqrt(A_COEFF_POTENTIAL / B_COEFF_POTENTIAL) if B_COEFF_POTENTIAL > 0 else PARTICLE_AMPLITUDE
vmax_plot = max(PARTICLE_AMPLITUDE, stable_phi_abs) * 1.1
vmin_plot = -vmax_plot * 0.5 # Allow for some negative wake

img = ax.imshow(simulator.phi, cmap='RdBu_r', 
                vmin=vmin_plot, vmax=vmax_plot, 
                animated=True, origin='lower',
                extent=[0, GRID_SIZE*DX, 0, GRID_SIZE*DX])

particle_marker, = ax.plot([], [], 'go', markersize=8, alpha=0.9, label="Field CoM")
ax.legend(loc='upper right')
cbar = plt.colorbar(img, ax=ax, label="Φ field amplitude", fraction=0.046, pad=0.04)

ax.set_xlabel("X position")
ax.set_ylabel("Y position")

log = {"t": [], "x": [], "y": [], "roughness": []}

def update_frame(frame_num):
    # --- physics update ----------------------------------------------------
    num_steps_per_frame = 10          # run extra PDE steps between frames
    for _ in range(num_steps_per_frame):
        simulator.step()

    current_phi = simulator.phi
    img.set_array(current_phi)

    # dynamic colour limits (bounded so they don't go crazy low/high)
    current_max = np.max(current_phi)
    current_min = np.min(current_phi)
    plot_vmax = max(current_max * 1.1, PARTICLE_AMPLITUDE * 0.5)
    plot_vmin = min(current_min * 0.8, -PARTICLE_AMPLITUDE * 0.2)
    if plot_vmax <= plot_vmin:
        plot_vmax = plot_vmin + 0.1
    img.set_clim(vmin=plot_vmin, vmax=plot_vmax)

    # ----------------------------------------------------------------------
    # --- quick probe: log COM & global roughness --------------------------
    # ----------------------------------------------------------------------
    # centre-of-mass of positive field (acts as harmony location)
    com_y, com_x = center_of_mass(np.maximum(0, current_phi))

    # roughness: mean magnitude of gradient of φ
    roughness = np.mean(np.abs(np.gradient(current_phi)))

    log["t"].append(simulator.time)
    log["x"].append(com_x)
    log["y"].append(com_y)
    log["roughness"].append(roughness)

    # ----------------------------------------------------------------------
    # update marker on plot
    if np.max(current_phi) > 0.05 * PARTICLE_AMPLITUDE:
        particle_marker.set_data([com_x * DX], [com_y * DX])
    else:
        particle_marker.set_data([], [])

    ax.set_title(f"Particle-Field Sailing on Substrate (Φ)\n"
                 f"Time: {simulator.time:.2f}")
    return img, particle_marker


ani = animation.FuncAnimation(fig, update_frame, frames=4000000, 
                              interval=30, blit=True, repeat=False) # Interval 30ms

plt.tight_layout()
plt.show()
df = pd.DataFrame(log)
df.head()        # quick peek
df.to_csv("com_roughness_log.csv", index=False)