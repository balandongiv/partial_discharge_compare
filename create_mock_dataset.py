"""
Script to create a mock dataset for partial discharge classification.

Creates:
- 10 samples with faultAnnotation=1 (fault present)
- 10 samples with faultAnnotation=0 (no fault)
- Synthetic PD-like signals in .npy format
- Annotation CSV matching the expected format
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
MOCK_DATASET_DIR = Path("mock_dataset")
STATION_ID = "mock001"
STATION_DIR = MOCK_DATASET_DIR / "contactless_pd_detection" / f"station_{STATION_ID}"
ANNOTATION_FILE = MOCK_DATASET_DIR / "inferred_annotation.csv"

# Signal parameters
SAMPLING_FREQ = 1_000_000  # 1 MHz sampling rate
SIGNAL_LENGTH = 10000  # 10 ms at 1 MHz (10000 samples)
BASE_STATION_ID = 99999  # Use a distinct station ID for mock data


def generate_pd_pulse(
    amplitude: float,
    center_freq: float,
    pulse_width: float,
    noise_level: float = 0.1,
    fs: float = SAMPLING_FREQ,
) -> np.ndarray:
    """
    Generate a synthetic partial discharge pulse.
    
    Args:
        amplitude: Peak amplitude of the pulse
        center_freq: Center frequency of the pulse (Hz)
        pulse_width: Width of the pulse in seconds
        noise_level: Standard deviation of additive noise
        fs: Sampling frequency (Hz)
    
    Returns:
        1D numpy array representing the PD pulse
    """
    t = np.arange(SIGNAL_LENGTH) / fs
    
    # Create a damped exponential pulse (typical PD pulse shape)
    pulse_start = SIGNAL_LENGTH // 4
    pulse_end = pulse_start + int(pulse_width * fs)
    
    signal = np.zeros(SIGNAL_LENGTH)
    
    if pulse_end < SIGNAL_LENGTH:
        pulse_t = t[pulse_start:pulse_end] - t[pulse_start]
        # Damped exponential with oscillation (typical PD characteristic)
        decay = np.exp(-pulse_t * 1e6)  # Fast decay
        oscillation = np.sin(2 * np.pi * center_freq * pulse_t)
        pulse = amplitude * decay * oscillation
        signal[pulse_start:pulse_end] = pulse
    
    # Add noise
    noise = np.random.normal(0, noise_level * amplitude, SIGNAL_LENGTH)
    signal += noise
    
    return signal.astype(np.float32)


def generate_fault_signal() -> np.ndarray:
    """
    Generate a signal with fault characteristics (stronger, more frequent PD pulses).
    
    Returns:
        1D numpy array with fault-like PD signal
    """
    # Multiple strong PD pulses
    signal = np.zeros(SIGNAL_LENGTH)
    
    # Add 3-5 strong PD pulses
    num_pulses = random.randint(3, 5)
    for _ in range(num_pulses):
        pulse_start = random.randint(1000, SIGNAL_LENGTH - 2000)
        pulse_width = random.uniform(0.0001, 0.0005)  # 0.1-0.5 ms
        amplitude = random.uniform(0.5, 1.5)  # Strong amplitude
        center_freq = random.uniform(100000, 5000000)  # 100 kHz - 5 MHz
        
        pulse = generate_pd_pulse(amplitude, center_freq, pulse_width, noise_level=0.15)
        # Place pulse at specific location
        pulse_len = len(pulse)
        if pulse_start + pulse_len <= SIGNAL_LENGTH:
            signal[pulse_start:pulse_start + pulse_len] += pulse[pulse_start:pulse_start + pulse_len]
    
    # Add background noise
    signal += np.random.normal(0, 0.05, SIGNAL_LENGTH)
    
    return signal.astype(np.float32)


def generate_normal_signal() -> np.ndarray:
    """
    Generate a signal without fault (weaker, fewer PD pulses or just noise).
    
    Returns:
        1D numpy array with normal/no-fault signal
    """
    # Weaker or no PD pulses
    signal = np.zeros(SIGNAL_LENGTH)
    
    # 50% chance of having a weak pulse, 50% chance of just noise
    if random.random() > 0.5:
        # Single weak pulse
        pulse_start = random.randint(2000, SIGNAL_LENGTH - 2000)
        pulse_width = random.uniform(0.00005, 0.0002)  # Shorter, weaker
        amplitude = random.uniform(0.1, 0.3)  # Weak amplitude
        center_freq = random.uniform(50000, 2000000)  # Lower frequency range
        
        pulse = generate_pd_pulse(amplitude, center_freq, pulse_width, noise_level=0.1)
        pulse_len = len(pulse)
        if pulse_start + pulse_len <= SIGNAL_LENGTH:
            signal[pulse_start:pulse_start + pulse_len] += pulse[pulse_start:pulse_start + pulse_len]
    
    # Add background noise (higher relative to signal)
    signal += np.random.normal(0, 0.1, SIGNAL_LENGTH)
    
    return signal.astype(np.float32)


def create_mock_dataset():
    """Create the complete mock dataset structure."""
    # Create directories
    STATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate annotation records
    annotations = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # 10 samples with fault (faultAnnotation=1)
    print("Generating 10 samples with fault annotations...")
    for i in range(1, 11):
        measurement_id = BASE_STATION_ID * 1000 + i
        signal = generate_fault_signal()
        filename = f"{measurement_id}.npy"
        filepath = STATION_DIR / filename
        np.save(filepath, signal)
        
        annotations.append({
            "idStation": BASE_STATION_ID,
            "idMeasurement": measurement_id,
            "faultAnnotation": 1,
            "timeStamp": (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"  Created {filename} (fault)")
    
    # 10 samples without fault (faultAnnotation=0)
    print("\nGenerating 10 samples without fault annotations...")
    for i in range(11, 21):
        measurement_id = BASE_STATION_ID * 1000 + i
        signal = generate_normal_signal()
        filename = f"{measurement_id}.npy"
        filepath = STATION_DIR / filename
        np.save(filepath, signal)
        
        annotations.append({
            "idStation": BASE_STATION_ID,
            "idMeasurement": measurement_id,
            "faultAnnotation": 0,
            "timeStamp": (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"  Created {filename} (no fault)")
    
    # Save annotation CSV
    df_annotations = pd.DataFrame(annotations)
    df_annotations.to_csv(ANNOTATION_FILE, index=False)
    print(f"\n[OK] Annotation file saved: {ANNOTATION_FILE}")
    print(f"  Total records: {len(annotations)}")
    print(f"  Fault samples (1): {len(df_annotations[df_annotations['faultAnnotation'] == 1])}")
    print(f"  Normal samples (0): {len(df_annotations[df_annotations['faultAnnotation'] == 0])}")
    
    print(f"\n[OK] Mock dataset created successfully!")
    print(f"  Location: {MOCK_DATASET_DIR}")
    print(f"  Station: {STATION_DIR}")
    print(f"  Signal files: {len(list(STATION_DIR.glob('*.npy')))}")


if __name__ == "__main__":
    create_mock_dataset()
