"""
Helper script to quickly determine RF amplitude and frequency for stability given the particle
properties and charge (defined below).
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi': 200, 'grid.linestyle': '--', 'grid.color': 'lightgrey'})

q_stab = 0.5 # Mathieu q parameter in the stability region
e = 1.602176634e-19 # unit charge in Coulombs

D_tube = 39.6e-3 # inner diameter of the CF nipple in m
d_elec = 3.175e-3 # diameter of an electrode in m (1/8" diameter rod)
r_0 = (d_elec/2.)/1.1464 # ratio that optimally matches ideal quadrupole field (Denison, J. Vac. Sci. Technol. 8, 266–269 (1971))
print(f'r_0 for Paul trap and main RF guide = {r_0*1e3:.3f} mm')

r_0_opt = 7.155815e-3 # 10.3e-3 # 7.1514462e-3 # trap radius for the Paul trap surrounding the optical trap
print(f'r_0 for electrodes surrounding optical trap = {r_0_opt*1e3:.3f} mm')

Q_e = 1e6 # number of charges
d_particle = 20e-6 # 166e-9 # diameter of the nanosphere/ethanol droplet in meters
r_particle = d_particle/2. # radius of the nanosphere/ethanol droplet in meters
V_particle = (4/3.)*np.pi*r_particle**3
rho_silica = 789 # 2.2e3 # density of silica/ethanol, kg/m^3
m_particle = rho_silica*V_particle # mass of a nanosphere in kg
amu_per_kg = 6.0221366516752e26
m_particle_amu = amu_per_kg * m_particle # mass of a nanosphere in amu (for SIMION)
print(f'Mass of the particle = {m_particle_amu:.3e}')

def V_rf(r_0, f, m=m_particle, Q=100, q=q_stab):
    return m*(2*np.pi*f)**2*r_0**2*q/(4*Q*e)

def f_rf(r_0, V, m=m_particle, Q=100, q=q_stab):
    return np.sqrt(V*2*(4*Q*e)/q/m/(2*np.pi)**2/r_0**2)

rf_freqs = np.logspace(0, 3, 1000)

fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
ax.loglog(rf_freqs, V_rf(r_0, rf_freqs, Q=Q_e), label='Paul trap/RF guide')
ax.fill_between(rf_freqs, V_rf(r_0, rf_freqs, Q=Q_e, q=0.2), V_rf(r_0, rf_freqs, Q=Q_e, q=0.7), alpha=0.3)
ax.loglog(rf_freqs, V_rf(r_0_opt, rf_freqs, Q=Q_e), label='Optical trap electrodes')
ax.fill_between(rf_freqs, V_rf(r_0_opt, rf_freqs, Q=Q_e, q=0.2), V_rf(r_0_opt, rf_freqs, Q=Q_e, q=0.7), alpha=0.3)
ax.set_xlabel('RF frequency [Hz]')
ax.set_ylabel('RF amplitude [V]')
ax.grid(which='both')
ax.legend()

plt.show()