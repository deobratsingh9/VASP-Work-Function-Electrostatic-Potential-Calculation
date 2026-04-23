import numpy as np
import matplotlib.pyplot as plt
import re

# ==========================================================
# ====== READ POSCAR (for exact z length) ==================
# ==========================================================
def read_poscar_z(filename="POSCAR"):
    with open(filename, 'r') as f:
        lines = f.readlines()

    scale = float(lines[1])
    lattice = np.array([list(map(float, lines[i].split())) for i in range(2, 5)])

    c_length = np.linalg.norm(lattice[2]) * scale
    return c_length


# ==========================================================
# ====== READ LOCPOT =======================================
# ==========================================================
def read_locpot(filename, z_length):
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if len(lines[i].split()) == 3:
            try:
                nx, ny, nz = map(int, lines[i].split())
                grid_index = i
            except:
                continue

    data = []
    for line in lines[grid_index + 1:]:
        data.extend([float(x) for x in line.split()])

    data = np.array(data)
    data = data.reshape((nx, ny, nz), order='F')

    planar_avg = data.mean(axis=(0, 1))
    z = np.linspace(0, z_length, nz)

    return z, planar_avg


# ==========================================================
# ====== READ .dat FILE ====================================
# ==========================================================
def read_dat_file(filename):
    distance = []
    potential = []

    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.split()
            if len(parts) >= 2:
                distance.append(float(parts[0]))
                potential.append(float(parts[1]))

    return np.array(distance), np.array(potential)


# ==========================================================
# ====== GET FERMI ENERGY ==================================
# ==========================================================
def get_fermi_energy(outcar="OUTCAR"):
    with open(outcar, 'r') as f:
        for line in f:
            if "E-fermi" in line:
                match = re.search(r"E-fermi\s*:\s*([0-9.+-Ee]+)", line)
                if match:
                    return float(match.group(1))
    return None


# ==========================================================
# ====== INPUT SELECTION ===================================
# ==========================================================
z_length = read_poscar_z("POSCAR")

# ---- USE ONE ONLY ----
# distance, potential = read_dat_file("planar_potential.dat")
distance, potential = read_locpot("LOCPOT", z_length)


# ==========================================================
# ====== FERMI SHIFT =======================================
# ==========================================================
fermi = get_fermi_energy("OUTCAR")

if fermi is None:
    raise ValueError("Fermi energy not found in OUTCAR")

potential_shifted = potential - fermi
vacuum_level = np.max(potential_shifted)
work_function = vacuum_level


# ==========================================================
# ====== PLOT (MANUSCRIPT QUALITY) =========================
# ==========================================================

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.5
})

plt.figure(figsize=(7,5))

# Main curve
plt.plot(distance, potential_shifted,
         color='black',
         linewidth=2.5,
         label="Planar Potential")

plt.xlim(0, 15)

# Fermi level
plt.axhline(0,
            linestyle='--',
            color='blue',
            linewidth=2,
            label="Fermi Level (0 eV)")

# Vacuum level
plt.axhline(vacuum_level,
            linestyle='--',
            color='red',
            linewidth=2,
            label=f"Vacuum = {vacuum_level:.3f} eV")

# Work function line
z_mid = distance[len(distance)//2]

plt.plot([z_mid, z_mid],
         [0, vacuum_level],
         color='darkorange',
         linewidth=3)

# Work function label
plt.text(z_mid + 0.3,
         vacuum_level/2,
         r"$\Phi$ = {:.3f} eV".format(work_function),
         rotation=90,
         fontsize=12,
         va='center',
         color='darkorange')

# Vacuum value label
plt.text(0.5,
         vacuum_level + 0.25,
         f"{vacuum_level:.3f} eV",
         color='red',
         fontsize=11)

# Labels
plt.xlabel("z (Å)", fontsize=13)
plt.ylabel("Electrostatic Potential (eV)", fontsize=13)

# Ticks
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

# NO GRID (clean manuscript style)
plt.grid(False)

# Legend
plt.legend(frameon=True,
           fontsize=11,
           edgecolor='black')

plt.tight_layout()

# Save high-quality image
plt.savefig("work_function_manuscript.png",
            dpi=600,
            bbox_inches='tight')

plt.show()


# ==========================================================
# ====== OUTPUT ============================================
# ==========================================================
print(f"Fermi Energy (original): {fermi:.4f} eV")
print(f"Vacuum Level (shifted): {vacuum_level:.4f} eV")
print(f"Work Function: {work_function:.4f} eV")
