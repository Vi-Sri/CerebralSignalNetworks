import numpy as np
import matplotlib.pyplot as plt

def generate_eeg_data(num_channels, num_samples, sampling_rate=1000):

    # Step 1: Generate Gaussian noise
    num_channels = 128
    num_samples = 500
    gaussian_noise = np.random.normal(0, 1, size=(num_channels, num_samples))

    # Example: Add a sinusoidal oscillation to each channel
    sampling_rate = 1000  # Assuming a 1000 Hz sampling rate for EEG
    frequency = 40  # Frequency of the oscillation (in Hz)
    time = np.arange(num_samples) / sampling_rate
    amplitude = 0.5  # Adjust the amplitude as needed
    eeg_data = gaussian_noise + amplitude * np.sin(2 * np.pi * frequency * time)


    return eeg_data

# Generate EEG-like data
num_channels = 128
num_samples = 500
eeg_data = generate_eeg_data(num_channels, num_samples)
print(eeg_data.shape)
# Plot a sample channel
channel_to_plot = 5
plt.figure(figsize=(10, 4))
plt.plot(eeg_data[channel_to_plot])
plt.title(f"Synthetic EEG-Like Signal (Channel {channel_to_plot + 1})")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()