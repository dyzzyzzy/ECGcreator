# Please Read README.md before use
# ------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import find_peaks
from matplotlib.widgets import RadioButtons

# -------------------------
# Settings
# -------------------------
fs = 250
duration = 10.0
t = np.arange(0, duration, 1 / fs)
rng = np.random.default_rng()

# Profile = HR-BEREICH + RR-Variabilität
profiles = {
    "Normal":      {"hr_range": (60, 100),  "rr_var": 0.03},
    "Bradykardie": {"hr_range": (30, 60),   "rr_var": 0.03},
    "Tachykardie": {"hr_range": (100, 160), "rr_var": 0.04},
    "Arrhythmisch":{"hr_range": (60, 120),  "rr_var": 0.18},
}

# -------------------------
# Filter functions
# -------------------------
def highpass(x, fg, cutoff=0.5, order=4):
    nyq = 0.5 * fg
    w = cutoff / nyq
    b, a = signal.butter(order, w, btype="highpass")
    return signal.filtfilt(b, a, x)

def lowpass(x, fg, cutoff=40.0, order=4):
    nyq = 0.5 * fg
    w = cutoff / nyq
    b, a = signal.butter(order, w, btype="lowpass")
    return signal.filtfilt(b, a, x)

def notch50(x, fg, q=30):
    w0 = 50 / (0.5 * fg)
    b, a = signal.iirnotch(w0, q)
    return signal.filtfilt(b, a, x)

# -------------------------
# ECG generator (PQRST + artefacts + RR jitter)
# -------------------------
def generate_ecg(profile_name: str):
    cfg = profiles[profile_name]
    hr_min, hr_max = cfg["hr_range"]
    hr = int(rng.integers(hr_min, hr_max + 1))   # zufällige HR im Bereich
    rr_var = float(cfg["rr_var"])
    base_rr = 60.0 / hr

    ecg = np.zeros_like(t)
    beat = 0.6  # starte später, damit P-Welle nicht "vor 0" liegt

    while beat < duration:
        # RR schwankt pro Schlag
        real_rr = base_rr * rng.normal(1.0, rr_var)
        real_rr = np.clip(real_rr, base_rr * 0.6, base_rr * 1.6)

        # R-Breite pro Schlag (immer positiv & sinnvoll)
        r_sigma = rng.normal(0.015, 0.003)
        r_sigma = np.clip(r_sigma, 0.008, 0.03)

        # PQRST (vereinfacht)
        ecg += 0.15 * np.exp(-0.5 * ((t - (beat - 0.2)) / 0.04) ** 2)     # P
        ecg += -0.2 * np.exp(-0.5 * ((t - (beat - 0.05)) / 0.01) ** 2)    # Q
        ecg += 1.0 * np.exp(-0.5 * ((t - beat) / r_sigma) ** 2)           # R
        ecg += -0.25 * np.exp(-0.5 * ((t - (beat + 0.05)) / 0.01) ** 2)   # S
        ecg += 0.15 * np.exp(-0.5 * ((t - (beat + 0.2)) / 0.08) ** 2)     # T

        beat += real_rr

    # Artefakte
    ecg += 0.1 * np.sin(2 * np.pi * 0.25 * t)          # Baseline Drift
    ecg += 0.03 * np.sin(2 * np.pi * 50 * t)           # 50 Hz Netzbrummen
    ecg += 0.05 * rng.standard_normal(len(t))          # Rauschen

    return hr, ecg

# -------------------------
# Analysis helpers
# -------------------------
def compute_bpm_and_peaks(ecg_filt):
    peaks, _ = find_peaks(
        ecg_filt,
        height=0.75,                 
        distance=int(0.3 * fs)       # ~300ms damit man mögliche doppelte peaks nicht liest
    )

    if len(peaks) >= 2:
        rr_sec = np.diff(peaks) / fs
        bpm = 60.0 / np.median(rr_sec)
    else:
        bpm = np.nan

    if np.isfinite(bpm):
        if bpm < 60:
            symptom = "Bradykardie"
        elif bpm <= 100:
            symptom = "normales Herzverhalten"
        else:
            symptom = "Tachykardie"
    else:
        symptom = "keine zuverlässige BPM"

    return peaks, bpm, symptom

# -------------------------
# UI / Plot setup
# -------------------------
fig = plt.figure(figsize=(12, 5), dpi=120)

ax = fig.add_axes([0.08, 0.14, 0.72, 0.78])      # main plot
ax_menu = fig.add_axes([0.83, 0.50, 0.15, 0.35]) # menu
ax_menu.set_title("Profil")

# Bottom info text area
info_text = fig.text(0.08, 0.04, "", fontsize=11)

radio = RadioButtons(ax_menu, list(profiles.keys()), active=0)

def update_plot(profile_name):
    # 1) Generate raw ECG
    chosen_hr, ecg_raw = generate_ecg(profile_name)

    # 2) Filter chain
    ecg_filt = highpass(ecg_raw, fs, cutoff=0.5)
    ecg_filt = notch50(ecg_filt, fs, q=30)
    ecg_filt = lowpass(ecg_filt, fs, cutoff=40.0)

    # 3) Peaks + BPM + label
    peaks, bpm, symptom = compute_bpm_and_peaks(ecg_filt)

    # 4) Plot
    ax.clear()
    ax.grid(alpha=0.2)

    ax.plot(t, ecg_raw, label="Raw", alpha=0.35, linewidth=1.5)
    ax.plot(t, ecg_filt, label="Filtered", linewidth=2.2)

    ax.plot(
        t[peaks], ecg_filt[peaks],
        "ro", markersize=8, markerfacecolor="none",
        label="R-Peaks"
    )

    # Optional: numbering
    for i, p in enumerate(peaks):
        ax.text(t[p], ecg_filt[p] + 0.05, str(i + 1), color="red")

    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Profil: {profile_name} | HR gewählt: {chosen_hr} BPM | gemessen: {bpm:.1f} BPM")
    ax.legend(loc="upper right")

    info_text.set_text(f"Dieses ECG deutet auf {symptom} hin")

    fig.canvas.draw_idle()

radio.on_clicked(update_plot)

# initial draw
update_plot("Normal")

plt.show()
