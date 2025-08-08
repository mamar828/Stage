import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from astropy.io import ascii

# ========== PARAMÈTRES ==========
N = 48  # nombre de canaux
x = np.arange(N)
center = N // 2
sigma_input = 3.5
sigma_lsf = 2.20
noise_amplitude = 1e-2
iterations = 50

# ========== CRÉATION DES FICHIERS NON CENTRÉS / NON NORMALISÉS ==========
raw_input = np.exp(-0.5 * ((x - (center + 5)) / sigma_input) ** 2)
raw_input += noise_amplitude * np.random.normal(size=N)

raw_lsf = np.exp(-0.5 * ((x - (center - 4)) / sigma_lsf) ** 2)

ascii.write({'input': raw_input}, 'input.txt', overwrite=True)
ascii.write({'lsf': raw_lsf}, 'lsf.txt', overwrite=True)

# ========== LECTURE DES FICHIERS + CENTRAGE & NORMALISATION ==========
input_signal = ascii.read('input.txt')['input']
lsf = ascii.read('lsf.txt')['lsf']

def recenter_profile(profile, center=center):
    x_vals = np.arange(len(profile))
    centroid = np.sum(profile * x_vals) / np.sum(profile)
    shift = int(round(center - centroid))
    profile_shifted = np.roll(profile, shift)
    return profile_shifted

def normalize_area(signal):
    return signal / np.sum(signal)

input_signal = recenter_profile(input_signal)
input_signal = normalize_area(input_signal)

lsf = recenter_profile(lsf)
lsf = normalize_area(lsf)

# ========== DECONVOLUTION ==========
def richardson_lucy(image, psf, iterations=30, eps=1e-6):
    image = image.astype(np.float64)
    estimate = np.full_like(image, 0.5)
    psf_mirror = psf[::-1]
    for _ in range(iterations):
        conv = convolve(estimate, psf, mode='same')
        conv = np.clip(conv, eps, None)
        ratio = image / conv
        correction = convolve(ratio, psf_mirror, mode='same')
        estimate *= correction
        estimate = np.clip(estimate, 0, 1e3)
    return estimate

output_signal = richardson_lucy(input_signal, lsf, iterations)
output_signal = normalize_area(output_signal)
output_signal = recenter_profile(output_signal)
output_signal = normalize_area(output_signal)

# ========== RECONVOLUTION ==========
reconvolved = convolve(output_signal, lsf, mode='same')
reconvolved = normalize_area(reconvolved)
reconvolved = recenter_profile(reconvolved)

# ========== RÉSIDU ==========
residual = input_signal - reconvolved

# ========== MESURES DE DISPERSION ==========
def measure_sigma(signal):
    centroid = np.sum(signal * x) / np.sum(signal)
    var = np.sum(signal * (x - centroid) ** 2) / np.sum(signal)
    return np.sqrt(var)

sigma_in_meas = measure_sigma(input_signal)
sigma_out_meas = measure_sigma(output_signal)
sigma_lsf_meas = measure_sigma(lsf)
sigma_check = np.sqrt(sigma_out_meas**2 + sigma_lsf_meas**2)

# ========== NORMALISATION POUR PLOT ==========
def normalize_for_plot(signal):
    max_abs = np.max(np.abs(signal))
    return signal / max_abs if max_abs != 0 else signal

# ========== CENTRAGE X POUR TRACÉ ==========
x_centered = x - center

# ========== TRACÉ ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig.subplots_adjust(hspace=0)

# Tracé principal
ax1.plot(x_centered, normalize_for_plot(input_signal), label=f'Input (σ={sigma_in_meas:.2f})')
ax1.plot(x_centered, normalize_for_plot(lsf), label=f'LSF (σ={sigma_lsf_meas:.2f})')
ax1.plot(x_centered, normalize_for_plot(output_signal), label=f'Deconvolved (σ={sigma_out_meas:.2f})')
ax1.plot(x_centered, normalize_for_plot(reconvolved), linestyle='--', label='Reconvolved')
ax1.axvline(0, color='k', linestyle=':', label='Centre')
ax1.set_ylabel('Amplitude normalisée')

# Limites automatiques Y
all_signals = 1.05 * np.array([
    normalize_for_plot(input_signal),
    normalize_for_plot(lsf),
    normalize_for_plot(output_signal),
    normalize_for_plot(reconvolved)
])
ax1.set_ylim(np.min(all_signals), np.max(all_signals))

ax1.set_xticks(np.arange(-center, center + 1, 4))
ax1.legend(loc='upper right')
ax1.tick_params(labelbottom=False)
ax1.set_title(f'Deconvolution – Vérification: σ_in ≈ √(σ_out² + σ_lsf²) = {sigma_check:.2f} (LSF σ={sigma_lsf_meas:.2f})')

# Axes secondaires
ax1.secondary_xaxis('top').set_xticks(np.arange(-center, center + 1, 4))
ax1.secondary_xaxis('top').set_xlabel('Canal centré')
ax1.secondary_yaxis('right').set_ylabel('Amplitude normalisée')

# Tracé résidu
ax2.plot(x_centered, residual, color='gray')
ax2.axhline(0, color='k', linestyle='--')
ax2.set_ylabel('Résidu')
ax2.set_xlabel('Canal centré')
ax2.set_xticks(np.arange(-center, center + 1, 4))

mu_res = np.mean(residual)
std_res = np.std(residual)
ax2.set_title(f'Residu: μ = {mu_res:.3e}, σ = {std_res:.3e}')

plt.show()
