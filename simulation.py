import numpy as np
import emodeconnection as emc

# Simulation parameters
wavelength = 1550          # nm
dx, dy = 10, 10            # nm resolution
window_width = 600 + 2*800 # nm (core + trenches)
num_modes = 2

# Layer thicknesses and nanowire dimensions (nm)
h_au            = 100
h_bottom_sio2   = 177
h_nbn           = 10    # NbN thickness
w_nbn           = 100  # NbN width
h_si_thin       = 25
h_p_sio2        = 106
h_si_core2      = 195
h_sio2_grating  = 147
h_sio2_top      = 150
h_al            = 150

# Connect and set up EMode
em = emc.EMode(simulation_name='snsdpd_test')


em.add_material(
    name='NbN',
    refractive_index_equation='5.23 + 5.82j',  # complex index of NbN at 1550 nm
    wavelength_unit='nm',
    wavelength_range=[1500, 1600],  # valid range for the above index
    citation='Approximate NbN optical constants at 1550 nm'
)

em.settings(
    wavelength=wavelength,
    x_resolution=dx,
    y_resolution=dy,
    window_width=window_width,
    window_height=(h_au + h_bottom_sio2 + h_nbn + h_si_thin +
                   h_p_sio2 + h_si_core2 + h_sio2_grating +
                   h_sio2_top + h_al +200), # extra space for PML 
    pml_NSEW_bool=[1,1,1,1],
    num_pml_layers=20,
    num_modes=num_modes,
    background_material='Air',
    boundary_condition='00',  # no symmetry; TE/TM modes identified automatically
    simulation_name='snsdpd_test'
)

# Draw the stack (bottom to top)
em.shape(name='Au_layer',       material='Au',   height=h_au)
em.shape(name='bottom_SiO2',    material='SiO2', height=h_bottom_sio2)
em.shape(name='NbN_nanowire',   material='NbN',  width=w_nbn, height=h_nbn)
em.shape(name='thin_Si',        material='Si',   height=h_si_thin)
em.shape(name='P_SiO2',         material='SiO2', height=h_p_sio2)
em.shape(name='second_Si',      material='Si',   height=h_si_core2)
em.shape(name='SiO2_grating',   material='SiO2', height=h_sio2_grating)
em.shape(name='top_SiO2',       material='SiO2', height=h_sio2_top)
# Simple Al over‑layer; for a periodic grating define mask/mask_offset here
em.shape(name='Al_SWG',         material='Al',   height=h_al)

#breakpoint()
# Solve for the eigenmodes
try:
    em.FDM()
except Exception as e:
    print("EModeError details:", e)
    raise  # the FDM solver calculates effective index, loss and TE fraction [oai_citation:2‡docs.emodephotonix.com](https://docs.emodephotonix.com/emodeguide/solver-analysis.html#:~:text=Calculate%20the%20waveguide%20modes%20using,normalized%20to%20the%20equation%20below)

# Display a summary of mode index, TE fraction and loss
em.report()

# Plot fields and refractive index profiles if desired
em.plot()

# Compute absorption efficiency for a detector length of 20 µm
length_um  = 20
length_m   = length_um * 1e-6
loss_dB_per_m = np.array(em.get('loss_dB_per_m'))
TE_indices = em.get('TE_indices')  # indices of TE‑like modes [oai_citation:3‡docs.emodephotonix.com](https://docs.emodephotonix.com/emodeguide/solver-analysis.html#:~:text=It%20also%20calculates%20the%20TE,TM_indices)
TM_indices = em.get('TM_indices')  # indices of TM‑like modes

# Convert propagation loss to absorption efficiency: η = 1 − 10^(−α·L/10)
abs_te = 1 - 10 ** (-loss_dB_per_m[TE_indices[0]] * length_m / 10)
abs_tm = 1 - 10 ** (-loss_dB_per_m[TM_indices[0]] * length_m / 10)
extinction_ratio = abs_te / abs_tm
print(f'TE absorption efficiency: {abs_te:.3f}')
print(f'TM absorption efficiency: {abs_tm:.3f}')
print(f'Extinction ratio (TE/TM): {extinction_ratio:.3f}')

em.close( save = False )  # close EMode session without saving the project