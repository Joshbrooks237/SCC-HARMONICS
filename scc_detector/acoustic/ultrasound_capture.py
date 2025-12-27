"""
Ultrasound Capture System for SCC Detection

Multi-frequency ultrasound acquisition across:
- Surface acoustic (40-200 kHz): Texture and surface mapping
- Clinical ultrasound (5-15 MHz): Tissue penetration
- High-frequency research (20-50 MHz): Cellular-level resolution

Hardware interfaces for:
- Murata ultrasonic transducers (40 kHz)
- Butterfly iQ+ (clinical)
- Research-grade high-frequency systems
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time


class UltrasoundMode(Enum):
    """Ultrasound operating modes"""
    SURFACE_ACOUSTIC = "surface"      # 40-200 kHz
    CLINICAL_BMODE = "clinical"        # 5-15 MHz
    HIGH_FREQUENCY = "high_freq"       # 20-50 MHz
    DOPPLER = "doppler"                # Blood flow
    HARMONIC = "harmonic"              # Harmonic imaging


@dataclass
class UltrasoundCapture:
    """Ultrasound data container"""
    # Raw signal data
    rf_data: np.ndarray  # Raw RF data [time x channels]
    envelope: np.ndarray  # Envelope-detected signal
    
    # Imaging parameters
    center_frequency: float  # Hz
    sampling_rate: float  # Hz
    mode: UltrasoundMode
    
    # Harmonic data
    harmonic_spectrum: Dict[int, np.ndarray] = field(default_factory=dict)  # Harmonic order -> amplitude
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    depth_mm: float = 10.0
    focus_depth_mm: float = 5.0
    gain_db: float = 40.0
    
    patient_id: str = ""
    body_location: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def get_bmode_image(self) -> np.ndarray:
        """Convert to B-mode image (log-compressed envelope)"""
        if self.envelope is None or len(self.envelope) == 0:
            return np.zeros((100, 100), dtype=np.uint8)
        
        # Log compression
        env_log = 20 * np.log10(self.envelope + 1e-10)
        
        # Normalize to 0-255
        env_min = np.percentile(env_log, 1)
        env_max = np.percentile(env_log, 99)
        normalized = np.clip((env_log - env_min) / (env_max - env_min + 1e-10) * 255, 0, 255)
        
        return normalized.astype(np.uint8)


class UltrasoundHardwareInterface:
    """Interface for ultrasound hardware"""
    
    def __init__(self, simulation_mode: bool = True):
        """
        Initialize ultrasound hardware interface.
        
        Args:
            simulation_mode: Use simulated data if True
        """
        self.simulation_mode = simulation_mode
        self.audio_interface = None
        self.clinical_probe = None
        self.hf_probe = None
        
        self._initialize_hardware()
    
    def _initialize_hardware(self):
        """Initialize available hardware"""
        if self.simulation_mode:
            print("🔊 Ultrasound system in simulation mode")
            return
        
        # Try to initialize audio interface for surface acoustic
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            print(f"✓ Audio interface detected: {len(devices)} devices")
            self.audio_interface = sd
        except ImportError:
            print("⚠️  sounddevice not installed")
        except Exception as e:
            print(f"⚠️  Audio interface error: {e}")
        
        # Clinical probe (Butterfly iQ) would need manufacturer SDK
        print("ℹ️  Clinical probe: Connect Butterfly iQ for clinical imaging")
        
        # High-frequency probe would need specialized hardware
        print("ℹ️  HF probe: Connect research probe for 20-50 MHz imaging")
    
    def capture_surface_acoustic(self, 
                                 frequency: float = 40000,
                                 duration: float = 0.1,
                                 sampling_rate: float = 192000) -> UltrasoundCapture:
        """
        Capture surface acoustic data using ultrasonic transducers.
        
        Args:
            frequency: Transmit frequency in Hz (40-200 kHz)
            duration: Capture duration in seconds
            sampling_rate: ADC sampling rate
            
        Returns:
            UltrasoundCapture with RF data
        """
        if self.simulation_mode:
            return self._simulate_surface_acoustic(frequency, duration, sampling_rate)
        
        # Real hardware capture
        samples = int(duration * sampling_rate)
        
        # Generate transmit pulse
        t = np.arange(samples) / sampling_rate
        tx_pulse = np.sin(2 * np.pi * frequency * t) * np.exp(-t / 0.001)  # Damped sine
        
        # TODO: Interface with actual audio hardware for real capture
        # For now, return simulated data
        return self._simulate_surface_acoustic(frequency, duration, sampling_rate)
    
    def _simulate_surface_acoustic(self, 
                                   frequency: float,
                                   duration: float,
                                   sampling_rate: float) -> UltrasoundCapture:
        """Simulate surface acoustic capture"""
        samples = int(duration * sampling_rate)
        t = np.arange(samples) / sampling_rate
        
        # Simulate reflected signal with harmonics
        signal = np.zeros(samples)
        
        # Fundamental
        signal += 1.0 * np.sin(2 * np.pi * frequency * t)
        
        # Add harmonics (non-linear tissue response)
        harmonic_amplitudes = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
        for i, amp in enumerate(harmonic_amplitudes, start=2):
            if i * frequency < sampling_rate / 2:  # Nyquist
                signal += amp * np.sin(2 * np.pi * i * frequency * t)
        
        # Add noise
        signal += np.random.normal(0, 0.05, samples)
        
        # Simulate depth-dependent attenuation
        depth_attenuation = np.exp(-t * 50)  # Exponential decay
        signal *= depth_attenuation
        
        # Envelope detection
        from scipy.signal import hilbert
        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        
        # Extract harmonic spectrum
        harmonic_spectrum = self._extract_harmonics(signal, frequency, sampling_rate)
        
        return UltrasoundCapture(
            rf_data=signal.reshape(-1, 1),
            envelope=envelope.reshape(-1, 1),
            center_frequency=frequency,
            sampling_rate=sampling_rate,
            mode=UltrasoundMode.SURFACE_ACOUSTIC,
            harmonic_spectrum=harmonic_spectrum,
            metadata={'simulated': True}
        )
    
    def capture_clinical(self, 
                        frequency_mhz: float = 10.0,
                        depth_mm: float = 30.0,
                        focus_mm: float = 15.0) -> UltrasoundCapture:
        """
        Capture clinical B-mode ultrasound.
        
        Args:
            frequency_mhz: Center frequency in MHz (5-15)
            depth_mm: Imaging depth
            focus_mm: Focus depth
            
        Returns:
            UltrasoundCapture with imaging data
        """
        if self.simulation_mode:
            return self._simulate_clinical(frequency_mhz, depth_mm, focus_mm)
        
        # Real clinical probe interface would go here
        return self._simulate_clinical(frequency_mhz, depth_mm, focus_mm)
    
    def _simulate_clinical(self, 
                          frequency_mhz: float,
                          depth_mm: float,
                          focus_mm: float) -> UltrasoundCapture:
        """Simulate clinical ultrasound B-mode image"""
        # Parameters
        c = 1540  # Speed of sound in tissue (m/s)
        frequency_hz = frequency_mhz * 1e6
        sampling_rate = 4 * frequency_hz  # 4x oversampling
        
        # Calculate scan dimensions
        lines = 128  # Number of scan lines
        samples_per_line = int(2 * depth_mm / 1000 / c * sampling_rate)
        
        # Initialize RF data
        rf_data = np.zeros((samples_per_line, lines))
        
        # Simulate tissue layers and lesion
        for line_idx in range(lines):
            t = np.arange(samples_per_line) / sampling_rate
            depth = c * t / 2 * 1000  # Depth in mm
            
            signal = np.zeros(samples_per_line)
            
            # Skin layer (around 2mm)
            skin_echo = np.exp(-((depth - 2) ** 2) / 0.5) * 0.8
            signal += skin_echo * np.sin(2 * np.pi * frequency_hz * t)
            
            # Subcutaneous tissue
            for d in np.arange(3, depth_mm, 2):
                tissue_echo = np.exp(-((depth - d) ** 2) / 0.3) * 0.2 * np.random.uniform(0.5, 1.5)
                signal += tissue_echo * np.sin(2 * np.pi * frequency_hz * t)
            
            # Lesion (if in center of image)
            if 40 < line_idx < 88:
                lesion_depth = 8  # mm
                lesion_echo = np.exp(-((depth - lesion_depth) ** 2) / 2) * 0.6
                signal += lesion_echo * np.sin(2 * np.pi * frequency_hz * t)
                
                # Lesion generates more harmonics (non-linear response)
                signal += lesion_echo * 0.3 * np.sin(2 * np.pi * 2 * frequency_hz * t)
            
            # Add speckle noise
            signal += np.random.normal(0, 0.1, samples_per_line)
            
            # Depth-dependent attenuation (0.5 dB/cm/MHz typical)
            attenuation_db = 0.5 * depth / 10 * frequency_mhz  # dB
            attenuation = 10 ** (-attenuation_db / 20)
            signal *= attenuation
            
            rf_data[:, line_idx] = signal
        
        # Envelope detection
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(rf_data, axis=0))
        
        # Extract harmonic content
        harmonic_spectrum = {}
        for line_idx in range(lines):
            line_harmonics = self._extract_harmonics(rf_data[:, line_idx], frequency_hz, sampling_rate)
            for order, amp in line_harmonics.items():
                if order not in harmonic_spectrum:
                    harmonic_spectrum[order] = []
                harmonic_spectrum[order].append(amp)
        
        # Average harmonics across lines
        for order in harmonic_spectrum:
            harmonic_spectrum[order] = np.mean(harmonic_spectrum[order])
        
        return UltrasoundCapture(
            rf_data=rf_data,
            envelope=envelope,
            center_frequency=frequency_hz,
            sampling_rate=sampling_rate,
            mode=UltrasoundMode.CLINICAL_BMODE,
            harmonic_spectrum=harmonic_spectrum,
            depth_mm=depth_mm,
            focus_depth_mm=focus_mm,
            metadata={'simulated': True, 'num_lines': lines}
        )
    
    def capture_high_frequency(self, 
                              frequency_mhz: float = 40.0,
                              depth_mm: float = 5.0) -> UltrasoundCapture:
        """
        Capture high-frequency ultrasound (20-50 MHz).
        
        Provides near-cellular resolution for superficial tissue.
        
        Args:
            frequency_mhz: Center frequency (20-50 MHz)
            depth_mm: Imaging depth (limited at high frequencies)
            
        Returns:
            UltrasoundCapture with high-resolution data
        """
        if self.simulation_mode:
            return self._simulate_high_frequency(frequency_mhz, depth_mm)
        
        return self._simulate_high_frequency(frequency_mhz, depth_mm)
    
    def _simulate_high_frequency(self, 
                                frequency_mhz: float,
                                depth_mm: float) -> UltrasoundCapture:
        """Simulate high-frequency ultrasound imaging"""
        c = 1540
        frequency_hz = frequency_mhz * 1e6
        sampling_rate = 4 * frequency_hz
        
        # High resolution
        lines = 256
        samples_per_line = int(2 * depth_mm / 1000 / c * sampling_rate)
        
        rf_data = np.zeros((samples_per_line, lines))
        
        for line_idx in range(lines):
            t = np.arange(samples_per_line) / sampling_rate
            depth = c * t / 2 * 1000
            
            signal = np.zeros(samples_per_line)
            
            # Epidermis layers (detailed)
            epidermis_layers = [0.1, 0.2, 0.3, 0.5, 0.8]  # mm
            for layer_depth in epidermis_layers:
                echo = np.exp(-((depth - layer_depth) ** 2) / 0.02) * 0.3
                signal += echo * np.sin(2 * np.pi * frequency_hz * t)
            
            # Dermis
            dermis_echo = np.exp(-((depth - 1.5) ** 2) / 0.5) * 0.4
            signal += dermis_echo * np.sin(2 * np.pi * frequency_hz * t)
            
            # SCC lesion in center
            if 80 < line_idx < 176:
                lesion_center_depth = 0.8  # mm
                lesion_echo = np.exp(-((depth - lesion_center_depth) ** 2) / 0.3) * 0.7
                signal += lesion_echo * np.sin(2 * np.pi * frequency_hz * t)
                
                # SCC signature: irregular structure, multiple harmonics
                for harm in [2, 3, 4]:
                    if harm * frequency_hz < sampling_rate / 2:
                        signal += lesion_echo * (0.2 / harm) * np.sin(2 * np.pi * harm * frequency_hz * t)
            
            # High attenuation at high frequency
            attenuation_db = 2.0 * depth / 10 * frequency_mhz  # Higher attenuation
            attenuation = 10 ** (-attenuation_db / 20)
            signal *= attenuation
            
            # Fine speckle
            signal += np.random.normal(0, 0.05, samples_per_line)
            
            rf_data[:, line_idx] = signal
        
        from scipy.signal import hilbert
        envelope = np.abs(hilbert(rf_data, axis=0))
        
        harmonic_spectrum = self._extract_harmonics(rf_data[:, lines//2], frequency_hz, sampling_rate)
        
        return UltrasoundCapture(
            rf_data=rf_data,
            envelope=envelope,
            center_frequency=frequency_hz,
            sampling_rate=sampling_rate,
            mode=UltrasoundMode.HIGH_FREQUENCY,
            harmonic_spectrum=harmonic_spectrum,
            depth_mm=depth_mm,
            metadata={'simulated': True, 'num_lines': lines}
        )
    
    def _extract_harmonics(self, 
                          signal: np.ndarray,
                          fundamental_freq: float,
                          sampling_rate: float,
                          max_harmonic: int = 8) -> Dict[int, float]:
        """
        Extract harmonic amplitudes from signal.
        
        Args:
            signal: Time-domain signal
            fundamental_freq: Fundamental frequency in Hz
            sampling_rate: Sampling rate in Hz
            max_harmonic: Maximum harmonic order to extract
            
        Returns:
            Dictionary mapping harmonic order to amplitude
        """
        # FFT
        n = len(signal)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1/sampling_rate)
        
        # Get positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        fft_pos = np.abs(fft[pos_mask])
        
        harmonics = {}
        
        for order in range(1, max_harmonic + 1):
            target_freq = order * fundamental_freq
            
            if target_freq >= sampling_rate / 2:
                break
            
            # Find nearest frequency bin
            idx = np.argmin(np.abs(freqs_pos - target_freq))
            
            # Average over nearby bins for robustness
            window = 5
            start = max(0, idx - window)
            end = min(len(fft_pos), idx + window)
            
            harmonics[order] = float(np.mean(fft_pos[start:end]))
        
        return harmonics
    
    def capture_full_spectrum(self) -> Dict[str, UltrasoundCapture]:
        """
        Capture across entire frequency spectrum.
        
        Returns:
            Dictionary with captures at different frequencies
        """
        print("\n" + "=" * 60)
        print("    FULL SPECTRUM ULTRASOUND ACQUISITION")
        print("    40 kHz → 50 MHz Multi-Frequency Scan")
        print("=" * 60)
        
        captures = {}
        
        # Surface acoustic frequencies
        print("\n[1/4] Surface Acoustic (40-200 kHz)...")
        for freq_khz in [40, 80, 120, 200]:
            print(f"    Capturing at {freq_khz} kHz...", end=" ")
            captures[f"surface_{freq_khz}kHz"] = self.capture_surface_acoustic(
                frequency=freq_khz * 1000
            )
            print("✓")
        
        # Clinical frequencies
        print("\n[2/4] Clinical Ultrasound (5-15 MHz)...")
        for freq_mhz in [5, 10, 15]:
            print(f"    Capturing at {freq_mhz} MHz...", end=" ")
            captures[f"clinical_{freq_mhz}MHz"] = self.capture_clinical(
                frequency_mhz=freq_mhz
            )
            print("✓")
        
        # High frequency
        print("\n[3/4] High-Frequency Research (20-50 MHz)...")
        for freq_mhz in [20, 30, 40, 50]:
            print(f"    Capturing at {freq_mhz} MHz...", end=" ")
            captures[f"highfreq_{freq_mhz}MHz"] = self.capture_high_frequency(
                frequency_mhz=freq_mhz
            )
            print("✓")
        
        print("\n[4/4] Harmonic analysis complete")
        print(f"✓ Captured {len(captures)} frequency bands")
        
        return captures

