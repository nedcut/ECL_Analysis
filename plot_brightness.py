import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_brightness(csv_path):
    df = pd.read_csv(csv_path)
    if 'brightness' not in df.columns:
        print("CSV must contain a 'brightness' column.")
        return

    brightness = df['brightness'].values
    frames = df['frame'].values if 'frame' in df.columns else np.arange(len(brightness))

    avg = np.mean(brightness)
    std = np.std(brightness)
    peak = np.max(brightness)
    peak_frame = frames[np.argmax(brightness)]

    plt.figure(figsize=(10, 6))
    plt.plot(frames, brightness, label='Brightness', color='blue')
    plt.axhline(avg, color='orange', linestyle='--', label=f'Average: {avg:.2f}')
    plt.axhline(avg + std, color='green', linestyle=':', label=f'+1 Std: {avg+std:.2f}')
    plt.axhline(avg - std, color='green', linestyle=':', label=f'-1 Std: {avg-std:.2f}')
    plt.scatter([peak_frame], [peak], color='red', zorder=5, label=f'Peak: {peak:.2f} (Frame {peak_frame})')

    plt.title(f'Brightness Over Time\n{os.path.basename(csv_path)}')
    plt.xlabel('Frame')
    plt.ylabel('Brightness')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_brightness.py <brightness_csv_file>")
        sys.exit(1)
    plot_brightness(sys.argv[1])
