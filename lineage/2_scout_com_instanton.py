import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import center_of_mass

class HybridFieldIntelligence:
    def __init__(self, grid_size=128):
        self.grid_size = grid_size
        self.dt = 0.02
        self.dx = 1.0
        self.time = 0.0
        
        # COM-Instanton parameters (wave field)
        self.c_wave = 1.0
        self.a_potential = 0.1
        self.b_potential = 0.1
        
        # Wave field state
        self.phi = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.phi_prev = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        # Attractor field (fractal landscape)
        self.attractor_field = self.generate_fractal_noise(grid_size, beta=1.2, seed=42)
        self.attractor_grad_x, self.attractor_grad_y = np.gradient(self.attractor_field)
        
        # Scout parameters
        self.scout_pos = np.array([grid_size/2, grid_size/2], dtype=float)
        self.scout_vel = np.zeros(2, dtype=float)
        self.scout_mass = 5.0
        self.scout_lr = 0.3
        self.noise_amp = 0.03
        
        # Coupling parameters
        self.wave_to_scout_coupling = 0.5
        self.scout_to_wave_coupling = 0.2
        
        # History tracking
        self.wave_com_history = []
        self.scout_history = []
        self.coupling_strength_history = []
        
        # Initialize wave instanton
        self.initialize_wave_particle()
        
    def generate_fractal_noise(self, size, beta=1.2, seed=None):
        """Generate 2D fractal noise"""
        if seed is not None:
            np.random.seed(seed)
        kx = fftfreq(size).reshape(1, -1)
        ky = fftfreq(size).reshape(-1, 1)
        k = np.sqrt(kx*kx + ky*ky)
        k[0,0] = 1e-6
        spectrum = (k**(-beta/2)) * np.exp(2j*np.pi*np.random.rand(size, size))
        noise = np.real(ifft2(spectrum))
        return (noise - noise.min()) / (noise.max() - noise.min())
    
    def initialize_wave_particle(self):
        """Initialize wave instanton"""
        center_x, center_y = self.grid_size * 0.3, self.grid_size * 0.5
        amplitude = 1.5
        radius = 8
        
        y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
        dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
        profile = amplitude / np.cosh(np.sqrt(dist_sq) / radius)
        self.phi += profile
        
        # Add initial velocity
        velocity_x, velocity_y = 0.2, 0.1
        phi_grad_x = (np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / (2 * self.dx)
        phi_grad_y = (np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / (2 * self.dx)
        phi_t_initial = -velocity_x * phi_grad_x - velocity_y * phi_grad_y
        self.phi_prev = self.phi - phi_t_initial * self.dt
    
    def laplacian(self, field):
        """Compute 2D Laplacian"""
        lap_x = (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / self.dx**2
        lap_y = (np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / self.dx**2
        return lap_x + lap_y
    
    def get_wave_com(self):
        """Get center of mass of wave field"""
        positive_phi = np.maximum(0, self.phi)
        if np.max(positive_phi) > 0.05:
            try:
                threshold = 0.2 * np.max(positive_phi)
                thresholded = np.where(positive_phi > threshold, positive_phi, 0)
                if np.sum(thresholded) > 1e-9:
                    com_y, com_x = center_of_mass(thresholded)
                    return np.array([com_x, com_y])
            except:
                pass
        # Fallback to peak
        peak_y, peak_x = np.unravel_index(np.argmax(positive_phi), positive_phi.shape)
        return np.array([peak_x, peak_y])
    
    def step(self):
        """One hybrid evolution step"""
        # 1. Evolve wave field (COM-instanton dynamics)
        lap_phi = self.laplacian(self.phi)
        wave_force = self.c_wave**2 * lap_phi
        potential_force = self.a_potential * self.phi - self.b_potential * self.phi**3
        
        # Get current wave COM for coupling
        wave_com = self.get_wave_com()
        
        # 2. Scout influences wave potential (scout -> wave coupling)
        scout_x, scout_y = int(self.scout_pos[0]), int(self.scout_pos[1])
        scout_x = np.clip(scout_x, 0, self.grid_size-1)
        scout_y = np.clip(scout_y, 0, self.grid_size-1)
        
        # Create localized influence from scout
        y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
        scout_influence = np.exp(-((x_coords - self.scout_pos[0])**2 + 
                                 (y_coords - self.scout_pos[1])**2) / (20**2))
        
        # Scout adds bias to wave potential
        modified_potential_force = (potential_force + 
                                  self.scout_to_wave_coupling * scout_influence * 
                                  self.attractor_field[scout_y, scout_x])
        
        # Evolve wave
        phi_new = (2 * self.phi - self.phi_prev + 
                  self.dt**2 * (wave_force + modified_potential_force))
        
        self.phi_prev = self.phi.copy()
        self.phi = phi_new
        
        # 3. Wave influences scout (wave -> scout coupling)
        # Scout feels both attractor gradient and wave field influence
        scout_x_int, scout_y_int = int(self.scout_pos[0]), int(self.scout_pos[1])
        scout_x_int = np.clip(scout_x_int, 0, self.grid_size-1)
        scout_y_int = np.clip(scout_y_int, 0, self.grid_size-1)
        
        # Attractor force
        attractor_force = np.array([
            self.attractor_grad_x[scout_y_int, scout_x_int],
            self.attractor_grad_y[scout_y_int, scout_x_int]
        ])
        
        # Wave field influence on scout
        wave_intensity = np.abs(self.phi[scout_y_int, scout_x_int])
        wave_gradient = np.array([
            (self.phi[scout_y_int, min(scout_x_int+1, self.grid_size-1)] - 
             self.phi[scout_y_int, max(scout_x_int-1, 0)]) / (2 * self.dx),
            (self.phi[min(scout_y_int+1, self.grid_size-1), scout_x_int] - 
             self.phi[max(scout_y_int-1, 0), scout_x_int]) / (2 * self.dx)
        ])
        
        # Combined force on scout
        total_force = (attractor_force + 
                      self.wave_to_scout_coupling * wave_intensity * wave_gradient)
        
        # Update scout
        fractal_noise = self.noise_amp * np.random.randn(2)
        self.scout_vel = (self.scout_vel + 
                         self.scout_lr * total_force / self.scout_mass + 
                         fractal_noise) * 0.95  # damping
        
        self.scout_pos += self.scout_vel
        self.scout_pos[0] = np.clip(self.scout_pos[0], 0, self.grid_size-1)
        self.scout_pos[1] = np.clip(self.scout_pos[1], 0, self.grid_size-1)
        
        # 4. Update history
        self.wave_com_history.append(wave_com.copy())
        self.scout_history.append(self.scout_pos.copy())
        coupling_strength = np.linalg.norm(wave_com - self.scout_pos)
        self.coupling_strength_history.append(coupling_strength)
        
        self.time += self.dt

# Create visualization
def create_hybrid_visualizer():
    hybrid = HybridFieldIntelligence()
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Hybrid COM-Instanton + Attractor-Scout Intelligence', fontsize=16, fontweight='bold')
    
    # Main field display
    ax_main = plt.subplot(2, 3, (1, 2))
    
    # Show attractor field as background
    attractor_img = ax_main.imshow(hybrid.attractor_field, 
                                  cmap='terrain', alpha=0.3, 
                                  origin='lower', extent=[0, 128, 0, 128])
    
    # Wave field overlay
    wave_img = ax_main.imshow(hybrid.phi, 
                             cmap='RdBu_r', alpha=0.8,
                             vmin=-1, vmax=2,
                             origin='lower', extent=[0, 128, 0, 128])
    
    # Markers
    wave_com_marker, = ax_main.plot([], [], 'yo', markersize=10, 
                                   label='Wave COM', markeredgecolor='black')
    scout_marker, = ax_main.plot([], [], 'ro', markersize=8, 
                                label='Scout', markeredgecolor='white')
    
    # Trails
    wave_trail, = ax_main.plot([], [], 'y-', alpha=0.6, linewidth=2)
    scout_trail, = ax_main.plot([], [], 'r-', alpha=0.6, linewidth=2)
    
    ax_main.set_title('Hybrid Field Dynamics')
    ax_main.set_xlabel('X Position')
    ax_main.set_ylabel('Y Position')
    ax_main.legend()
    
    # Wave field evolution
    ax_wave = plt.subplot(2, 3, 3)
    wave_evolution, = ax_wave.plot([], [], 'b-', linewidth=2)
    ax_wave.set_title('Wave Field Energy')
    ax_wave.set_xlabel('Time Steps')
    ax_wave.set_ylabel('Total Energy')
    ax_wave.grid(True)
    
    # Coupling strength
    ax_coupling = plt.subplot(2, 3, 4)
    coupling_line, = ax_coupling.plot([], [], 'g-', linewidth=2)
    ax_coupling.set_title('Wave-Scout Coupling Distance')
    ax_coupling.set_xlabel('Time Steps')
    ax_coupling.set_ylabel('Distance')
    ax_coupling.grid(True)
    
    # Phase space
    ax_phase = plt.subplot(2, 3, 5)
    wave_phase, = ax_phase.plot([], [], 'yo', alpha=0.6, markersize=4)
    scout_phase, = ax_phase.plot([], [], 'ro', alpha=0.6, markersize=4)
    ax_phase.set_title('Position Phase Space')
    ax_phase.set_xlabel('X Position')
    ax_phase.set_ylabel('Y Position')
    ax_phase.set_xlim(0, 128)
    ax_phase.set_ylim(0, 128)
    
    # Statistics
    ax_stats = plt.subplot(2, 3, 6)
    ax_stats.axis('off')
    stats_text = ax_stats.text(0.1, 0.9, '', transform=ax_stats.transAxes,
                              fontfamily='monospace', verticalalignment='top')
    
    # Animation data
    wave_energies = []
    max_history = 200
    
    def animate(frame):
        # Evolve system
        for _ in range(3):  # Multiple steps per frame
            hybrid.step()
        
        # Update wave field display
        wave_img.set_array(hybrid.phi)
        current_max = max(1.5, np.max(hybrid.phi))
        current_min = min(-0.5, np.min(hybrid.phi))
        wave_img.set_clim(vmin=current_min, vmax=current_max)
        
        # Update markers
        if len(hybrid.wave_com_history) > 0:
            wave_com = hybrid.wave_com_history[-1]
            wave_com_marker.set_data([wave_com[0]], [wave_com[1]])
            
        scout_pos = hybrid.scout_pos
        scout_marker.set_data([scout_pos[0]], [scout_pos[1]])
        
        # Update trails
        if len(hybrid.wave_com_history) > 1:
            recent_wave = hybrid.wave_com_history[-min(50, len(hybrid.wave_com_history)):]
            wave_x = [pos[0] for pos in recent_wave]
            wave_y = [pos[1] for pos in recent_wave]
            wave_trail.set_data(wave_x, wave_y)
            
        if len(hybrid.scout_history) > 1:
            recent_scout = hybrid.scout_history[-min(50, len(hybrid.scout_history)):]
            scout_x = [pos[0] for pos in recent_scout]
            scout_y = [pos[1] for pos in recent_scout]
            scout_trail.set_data(scout_x, scout_y)
        
        # Update energy plot
        wave_energy = np.sum(hybrid.phi**2)
        wave_energies.append(wave_energy)
        if len(wave_energies) > max_history:
            wave_energies.pop(0)
        
        wave_evolution.set_data(range(len(wave_energies)), wave_energies)
        if len(wave_energies) > 1:
            ax_wave.set_xlim(0, len(wave_energies))
            ax_wave.set_ylim(min(wave_energies) * 0.9, max(wave_energies) * 1.1)
        
        # Update coupling
        if len(hybrid.coupling_strength_history) > 0:
            recent_coupling = hybrid.coupling_strength_history[-min(max_history, len(hybrid.coupling_strength_history)):]
            coupling_line.set_data(range(len(recent_coupling)), recent_coupling)
            if len(recent_coupling) > 1:
                ax_coupling.set_xlim(0, len(recent_coupling))
                ax_coupling.set_ylim(0, max(recent_coupling) * 1.1)
        
        # Update phase space
        if len(hybrid.wave_com_history) > 0:
            recent_wave = hybrid.wave_com_history[-min(100, len(hybrid.wave_com_history)):]
            wave_x = [pos[0] for pos in recent_wave]
            wave_y = [pos[1] for pos in recent_wave]
            wave_phase.set_data(wave_x, wave_y)
            
        if len(hybrid.scout_history) > 0:
            recent_scout = hybrid.scout_history[-min(100, len(hybrid.scout_history)):]
            scout_x = [pos[0] for pos in recent_scout]
            scout_y = [pos[1] for pos in recent_scout]
            scout_phase.set_data(scout_x, scout_y)
        
        # Update statistics
        if len(hybrid.wave_com_history) > 0 and len(hybrid.scout_history) > 0:
            wave_com = hybrid.wave_com_history[-1]
            scout_pos = hybrid.scout_pos
            distance = np.linalg.norm(wave_com - scout_pos)
            
            stats_str = f"""HYBRID INTELLIGENCE STATS
Time: {hybrid.time:.2f}s

Wave COM: ({wave_com[0]:.1f}, {wave_com[1]:.1f})
Scout Pos: ({scout_pos[0]:.1f}, {scout_pos[1]:.1f})
Coupling Distance: {distance:.1f}

Wave Energy: {wave_energy:.2f}
Scout Velocity: {np.linalg.norm(hybrid.scout_vel):.3f}

Wave→Scout Coupling: {hybrid.wave_to_scout_coupling}
Scout→Wave Coupling: {hybrid.scout_to_wave_coupling}

EMERGENT BEHAVIORS:
{"🌊 Wave tunneling" if np.max(hybrid.phi) > 1.0 else ""}
{"🎯 Scout converging" if np.linalg.norm(hybrid.scout_vel) < 0.1 else ""}
{"🔄 Synchronized motion" if distance < 10 else ""}
{"⚡ High coupling" if distance < 5 else ""}
"""
            stats_text.set_text(stats_str)
        
        return (wave_img, wave_com_marker, scout_marker, wave_trail, scout_trail,
                wave_evolution, coupling_line, wave_phase, scout_phase, stats_text)
    
    ani = animation.FuncAnimation(fig, animate, frames=1000,
                                 interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    return fig, ani, hybrid

# Run the visualization
if __name__ == "__main__":
    print("🌌 Launching Hybrid COM-Instanton + Attractor-Scout Visualizer...")
    print("🌊 Yellow: Wave COM (quantum-like tunneling)")
    print("🔴 Red: Scout (classical goal-seeking)")
    print("🔄 Watch for emergent coupling behaviors!")
    
    fig, ani, hybrid = create_hybrid_visualizer()
    plt.show()