import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import mne

# Set random seed for reproducibility
np.random.seed(42)

def generate_test_signal(t):
    """Generate a test signal with multiple frequency components"""
    # Create a signal with 3 frequency components: 5Hz, 10Hz, and 15Hz
    signal_5hz = 2.0 * np.sin(2 * np.pi * 5 * t)
    signal_10hz = 1.5 * np.sin(2 * np.pi * 10 * t)
    signal_15hz = 0.5 * np.sin(2 * np.pi * 15 * t)
    
    # Add some noise
    noise = 0.5 * np.random.normal(size=len(t))
    
    # Combine signals
    combined_signal = signal_5hz + signal_10hz + signal_15hz + noise
    return combined_signal

def plot_signal_and_spectrum(t, signal, sampling_rate):
    """Plot original signal and its frequency spectrum"""
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original signal
    ax1.plot(t, signal)
    ax1.set_title('Original Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Compute FFT
    n = len(t)
    yf = fft(signal)
    xf = fftfreq(n, 1/sampling_rate)
    
    # Plot only positive frequencies
    positive_freq_mask = xf >= 0
    ax2.plot(xf[positive_freq_mask], 2.0/n * np.abs(yf[positive_freq_mask]))
    ax2.set_title('Frequency Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def compute_and_plot_psd(signal, sampling_rate):
    """Compute and plot Power Spectral Density using different methods"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Method 1: Periodogram
    f, Pxx = signal.periodogram(signal, fs=sampling_rate)
    ax1.semilogy(f, Pxx)
    ax1.set_title('Power Spectral Density (Periodogram)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power/Frequency')
    ax1.grid(True)
    
    # Method 2: Welch's method
    f, Pxx = signal.welch(signal, fs=sampling_rate, nperseg=1024)
    ax2.semilogy(f, Pxx)
    ax2.set_title('Power Spectral Density (Welch\'s method)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power/Frequency')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def apply_bandpass_filter(signal, sampling_rate, lowcut, highcut):
    """Apply bandpass filter to signal"""
    nyquist = sampling_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal)
    return filtered_signal

def plot_eeg_bands(signal, sampling_rate):
    """Plot signal filtered in different EEG frequency bands"""
    # Define EEG bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    
    # Create subplots
    fig, axs = plt.subplots(len(bands), 1, figsize=(12, 12))
    t = np.arange(len(signal)) / sampling_rate
    
    for i, (band_name, (low, high)) in enumerate(bands.items()):
        # Apply bandpass filter
        filtered = apply_bandpass_filter(signal, sampling_rate, low, high)
        
        # Plot
        axs[i].plot(t, filtered)
        axs[i].set_title(f'{band_name} band ({low}-{high} Hz)')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Amplitude')
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    sampling_rate = 128  # Hz (same as in our EEG data)
    duration = 5  # seconds
    t = np.arange(0, duration, 1/sampling_rate)
    
    # Generate test signal
    test_signal = generate_test_signal(t)
    
    # 1. Plot original signal and its spectrum
    print("1. Plotting original signal and its frequency spectrum...")
    plot_signal_and_spectrum(t, test_signal, sampling_rate)
    
    # 2. Compute and plot PSD
    print("\n2. Computing and plotting Power Spectral Density...")
    compute_and_plot_psd(test_signal, sampling_rate)
    
    # 3. Plot signal in different EEG frequency bands
    print("\n3. Plotting signal filtered in different EEG frequency bands...")
    plot_eeg_bands(test_signal, sampling_rate)
    
    # Additional experiments with real EEG data
    print("\n4. Loading and analyzing sample EEG data...")
    try:
        # Try to load actual EEG data if available
        from scipy.io import loadmat
        eeg_data = loadmat('eeg_record25.mat')
        # Add your EEG data analysis here
        print("EEG data loaded successfully!")
    except:
        print("Note: No EEG data file found. Using only synthetic test signals.")