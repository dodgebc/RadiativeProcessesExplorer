import numpy as np
import streamlit as st
import plotly.graph_objects as go
import physics

from physics import Blackbody, PAHSpectrum

st.title('Molecular Cloud CO Emission Line Modeling')

st.sidebar.header('Cloud Parameters')
T = st.sidebar.slider('Excitation temperature (K)', 5, 30, 8)
sigma_nu = 1e9*st.sidebar.slider('Doppler broadening (GHz)', 0.001, 0.1, 0.04, format="%.1e")
N = 10**(st.sidebar.slider('12CO Column Density log10(N/cm⁻²)', 10., 23., 12., format="%.1e"))
ratio = st.sidebar.slider('12CO/13CO ratio', 10, 100, 50)


# Define frequency grid
nu_min = 100e9
nu_max = 130e9
nu_grid = np.linspace(nu_min, nu_max, 5000)

# CO data
center_nu = {'12CO': 115.271e9, '13CO': 110.201e9} # Hz
A10 = {'12CO': 6.78e-8, '13CO': 6.73e-8} # s⁻¹
T_1 = {'12CO': 5.56, '13CO': 5.3} # K
B0 = {'12CO': 2.78, '13CO': 2.65} # K
column = {'12CO': N, '13CO': N/ratio}

for k in ['12CO', '13CO']:
    phi_nu = np.exp(-(nu_grid - center_nu[k])**2 / (2 * sigma_nu**2))

    # Einstein coefficients
    B10 = physics.c**2/(2*physics.h*center_nu[k]**3) * A10[k]
    B01 = 3 * B10

    # Level occupancy
    N_l = column[k] / np.sqrt(1 + (T/B0[k])**2)
    N_u = N_l * np.exp(-T_1[k]/T)

    # Optical depth and intensity
    tau_nu = physics.h * nu_grid / (4*np.pi) * (N_l*B01 - N_u*B10) * phi_nu
    I_nu = (1 - np.exp(-tau_nu)) * physics.bb(nu_grid, T)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nu_grid/1e9, 
        y=I_nu,
        mode='lines', 
        name='Spectrum',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title=f"{k} J=1->0 Emission Line",
        xaxis_title="Frequency (GHz)",
        yaxis_title="Intensity (erg s⁻¹ Hz⁻¹)",
        xaxis_type="log",
        # yaxis_type="log",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


st.markdown(r"""
This is a simple model of the 12CO J=1->0 emission one might expect from a molecular cloud.
""")
