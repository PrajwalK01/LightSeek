import lightkurve as lk
import matplotlib.pyplot as plt

# Download Kepler-17 — confirmed hot Jupiter
print("Downloading Kepler-17 light curve...")
result = lk.search_lightcurve("Kepler-17", mission="kepler", quarter=1)
lc = result[0].download()

# Clean it
lc = lc.remove_nans().remove_outliers().normalize()

# Plot it
lc.plot(title="Kepler-17 — First Real Light Curve 🪐")
plt.savefig("results/plots/kepler17_first.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"Time range: {lc.time.value[0]:.2f} — {lc.time.value[-1]:.2f} days")
print(f"Total data points: {len(lc.flux)}")
print(f"Flux min: {lc.flux.value.min():.6f}")
print(f"Flux max: {lc.flux.value.max():.6f}")
print("✓ Done! Check results/plots/kepler17_first.png")