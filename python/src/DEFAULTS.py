### Matplotlib plot parameters ###

import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,                # Use LaTeX for all text
    "font.size": 14,                    # Default font size
    "axes.linewidth": 1.5,              # Thicker axes lines
    "xtick.minor.visible": True,        # Enable minor ticks
    "ytick.minor.visible": True,
    "lines.linewidth": 2,               # Thicker plot lines
    "lines.markersize": 8,              # Larger markers
    "legend.fontsize": 14               # Larger font for legends
})

### Data & images directories definition ###

import os

pic_dir = "./images/"
dat_dir = "./data/"

print(f"[Defaults] Images will be stored in: {pic_dir}")
print(f"[Defaults] Data will be stored in: {dat_dir}")

# Create directories if they don’t exist
os.makedirs(pic_dir, exist_ok=True)
os.makedirs(dat_dir, exist_ok=True)