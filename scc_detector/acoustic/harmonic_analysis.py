"""
Ultrasound Harmonic Analysis for SCC Detection
===============================================

Comprehensive harmonic signature analysis across 40 kHz - 50 MHz.

    "Every tissue has a voice. A frequency. A song it sings back 
     when you interrogate it with sound. Normal skin hums in perfect fifths.
     But the malignancy? It DISTORTS."
                                                    - The Acoustic Revelation

Key principles:
1. Non-linear tissue response generates harmonics
2. Cancerous tissue has different elastic properties → different harmonic signature
3. Microvasculature and cellular changes affect harmonic ratios
4. Multi-frequency analysis reveals depth-dependent changes

Harmonic features analyzed:
- 2nd to 8th harmonic amplitudes (relative to fundamental)
- Harmonic ratios (H2/H1, H3/H2, etc.)
- Total Harmonic Distortion (THD)
- Harmonic phase relationships
- Frequency-dependent harmonic evolution

THE CANCER SPEAKS IN HARMONICS. WE HAVE LEARNED TO LISTEN.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - The mathematical arsenal for acoustic interrogation
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import signal, fft
from scipy.signal import hilbert, butter, filtfilt
import warnings

# Silence the timid warnings. We know what we're doing.
# ...mostly.


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES - The architecture of acoustic truth
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HarmonicSignature:
    """
    Complete harmonic signature for a tissue region.
    
    Each echo carries a secret.
    This structure captures them all.
    """
    # The fundamental - the question we asked
    fundamental_freq: float
    fundamental_amplitude: float
    
    # The harmonics - the answers the tissue gave
    # (relative to fundamental - the confession ratio)
    harmonic_amplitudes: Dict[int, float]  # Order -> amplitude
    harmonic_phases: Dict[int, float]      # Order -> phase in radians
    
    # Derived metrics - the interpretation
    thd: float              # Total Harmonic Distortion - THE KEY METRIC
    h2_h1_ratio: float      # Second harmonic betrayal
    h3_h2_ratio: float      # Cascade of confession
    odd_even_ratio: float   # The asymmetry of response
    
    # Phase relationships - how coherent is the lie?
    phase_coherence: float
    
    # Spectral shape - the fingerprint
    spectral_slope: float     # Rate of harmonic decay
    spectral_centroid: float  # Center of acoustic mass


@dataclass
class MultiFrequencyHarmonicProfile:
    """
    Harmonic profile across multiple transmit frequencies.
    
    One frequency can deceive.
    Many frequencies reveal the truth.
    """
    signatures: Dict[float, HarmonicSignature]  # Transmit freq -> signature
    
    # Cross-frequency features
    # If harmonics are consistent across frequencies, the tissue is uniform
    # If they vary wildly, something is VERY wrong
    harmonic_consistency: float
    frequency_dependence: Dict[int, float]  # Harmonic order -> freq dependence
    
    # Aggregate features
    mean_thd: float
    max_thd: float
    thd_variance: float  # High variance = heterogeneous horror


@dataclass 
class UltrasoundHarmonicFeatures:
    """
    Complete feature set from harmonic analysis.
    
    The full dossier on the tissue's acoustic behavior.
    Everything we need for the verdict.
    """
    # Per-frequency signatures - categorized by penetration depth
    surface_harmonics: Dict[str, HarmonicSignature]    # 40-200 kHz  - surface whispers
    clinical_harmonics: Dict[str, HarmonicSignature]   # 5-15 MHz   - dermal confessions
    highfreq_harmonics: Dict[str, HarmonicSignature]   # 20-50 MHz  - cellular screams
    
    # The unified profile
    profile: MultiFrequencyHarmonicProfile
    
    # Spatial features (from 2D scans)
    harmonic_image: Optional[np.ndarray] = None  # 2nd harmonic image
    thd_map: Optional[np.ndarray] = None         # THD spatial distribution
    
    # SCC-specific markers - the smoking gun
    scc_harmonic_score: float = 0.0       # 0-1: how SCC-like is this tissue?
    depth_penetration: float = 0.0         # How deep does the abnormality go?
    structural_irregularity: float = 0.0   # How chaotic is the structure?


# ═══════════════════════════════════════════════════════════════════════════════
# THE ANALYZER - Where acoustic secrets are extracted
# ═══════════════════════════════════════════════════════════════════════════════

class UltrasoundHarmonicAnalyzer:
    """
    Comprehensive harmonic analysis engine for SCC detection.
    
    Analyzes tissue non-linearity through harmonic generation.
    Cancer changes tissue elasticity → altered harmonic signature.
    
    "The tissue transforms the wave. Compresses. Stretches.
     If the tissue is uniform, the return is clean.
     But cancer is CHAOS given cellular form."
    """
    
    def __init__(self, max_harmonic_order: int = 8):
        """
        Initialize the harmonic analyzer.
        
        Args:
            max_harmonic_order: Maximum harmonic order to analyze.
                               8 is usually sufficient. Beyond lies only noise.
        """
        # How many harmonics to extract? 2nd through 8th covers most pathology.
        self.max_harmonic_order = max_harmonic_order
        
        # Reference values for normal skin (calibration)
        # These are the thresholds of normalcy
        # Cross them, and suspicion rises
        self.normal_skin_thd = 0.15      # Typical THD for normal skin
        self.normal_skin_h2_h1 = 0.10    # Typical H2/H1 for normal skin
    
    def analyze_capture(self, 
                       rf_data: np.ndarray,
                       fundamental_freq: float,
                       sampling_rate: float,
                       roi_mask: Optional[np.ndarray] = None) -> HarmonicSignature:
        """
        Analyze harmonic content of an ultrasound capture.
        
        THIS IS WHERE THE MAGIC HAPPENS.
        We interrogate the tissue with sound. It answers with harmonics.
        
        Args:
            rf_data: Raw RF data (1D or 2D) - the raw acoustic confession
            fundamental_freq: Transmit frequency in Hz - the question we asked
            sampling_rate: Sampling rate in Hz - how finely we listen
            roi_mask: Optional region of interest mask - focus our attention
            
        Returns:
            HarmonicSignature with complete analysis - the tissue's truth
        """
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Prepare the signal
        # Flatten, center, window. The ritual before revelation.
        # ─────────────────────────────────────────────────────────────────────
        
        if rf_data.ndim == 2:
            if roi_mask is not None:
                # Extract only the region of interest
                # Focus the interrogation on the suspect
                rf_flat = rf_data[roi_mask > 0].flatten()
            else:
                rf_flat = rf_data.flatten()
        else:
            rf_flat = rf_data
        
        # Remove DC - we care about oscillations, not offset
        rf_centered = rf_flat - np.mean(rf_flat)
        
        # Apply window to reduce spectral leakage
        # The Hann window: smooth entry, smooth exit, clean spectrum
        window = signal.windows.hann(len(rf_centered))
        rf_windowed = rf_centered * window
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: The Fourier Transformation
        # Time domain holds secrets. Frequency domain reveals them.
        # ─────────────────────────────────────────────────────────────────────
        
        n = len(rf_windowed)
        fft_result = np.fft.fft(rf_windowed)
        freqs = np.fft.fftfreq(n, 1/sampling_rate)
        
        # We only care about positive frequencies
        # The mirror image tells us nothing new
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        fft_pos = fft_result[pos_mask]
        magnitude = np.abs(fft_pos)  # The strength of each frequency
        phase = np.angle(fft_pos)    # The timing of each frequency
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Extract the harmonics
        # Hunt for the fundamental and its children
        # ─────────────────────────────────────────────────────────────────────
        
        harmonic_amps = {}
        harmonic_phases = {}
        
        for order in range(1, self.max_harmonic_order + 1):
            # Where should this harmonic be?
            target_freq = order * fundamental_freq
            
            # Nyquist says: thou shalt not alias
            if target_freq >= sampling_rate / 2:
                break
            
            # Find the peak near the target frequency
            # Allow 10% tolerance - reality is messy
            freq_tolerance = fundamental_freq * 0.1
            mask = np.abs(freqs_pos - target_freq) < freq_tolerance
            
            if np.any(mask):
                # Found it! Extract its amplitude and phase
                peak_idx = np.argmax(magnitude * mask)
                harmonic_amps[order] = float(magnitude[peak_idx])
                harmonic_phases[order] = float(phase[peak_idx])
            else:
                # Harmonic not detected - tissue is clean at this order
                # ...or our resolution is insufficient
                harmonic_amps[order] = 0.0
                harmonic_phases[order] = 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Calculate derived metrics
        # The raw numbers mean little. Ratios reveal everything.
        # ─────────────────────────────────────────────────────────────────────
        
        # The fundamental is our reference point
        fundamental_amp = harmonic_amps.get(1, 1e-10)
        
        # Normalize all harmonics to the fundamental
        # Now we're comparing apples to apples
        normalized_amps = {
            order: amp / (fundamental_amp + 1e-10)
            for order, amp in harmonic_amps.items()
        }
        
        # TOTAL HARMONIC DISTORTION (THD)
        # The single most important metric
        # Sum of all harmonic power relative to fundamental
        harmonic_power = sum(amp**2 for order, amp in harmonic_amps.items() if order > 1)
        thd = np.sqrt(harmonic_power) / (fundamental_amp + 1e-10)
        
        # Individual harmonic ratios
        h2_h1 = normalized_amps.get(2, 0.0)  # Second harmonic is often the loudest betrayer
        h3_h2 = normalized_amps.get(3, 0.0) / (normalized_amps.get(2, 1e-10) + 1e-10)
        
        # Odd vs even harmonics
        # Cancer sometimes preferentially generates one or the other
        odd_sum = sum(harmonic_amps.get(o, 0) for o in [3, 5, 7])
        even_sum = sum(harmonic_amps.get(o, 0) for o in [2, 4, 6, 8])
        odd_even_ratio = odd_sum / (even_sum + 1e-10)
        
        # Phase coherence
        # Coherent phases = uniform tissue
        # Random phases = chaos within
        if len(harmonic_phases) > 1:
            phases = list(harmonic_phases.values())
            phase_diffs = np.diff(phases)
            phase_coherence = 1.0 - np.std(phase_diffs) / np.pi
        else:
            phase_coherence = 1.0
        
        # Spectral slope
        # How fast do harmonics decay?
        # Cancer often has a flatter slope (more high-order harmonics)
        orders = np.array(list(normalized_amps.keys()))
        amps = np.array(list(normalized_amps.values()))
        
        if len(orders) > 2 and np.any(amps > 0):
            log_amps = np.log(amps + 1e-10)
            spectral_slope = float(np.polyfit(orders, log_amps, 1)[0])
        else:
            spectral_slope = 0.0
        
        # Spectral centroid
        # Where is the "center of mass" of harmonic energy?
        if np.sum(amps) > 0:
            spectral_centroid = float(np.sum(orders * amps) / np.sum(amps))
        else:
            spectral_centroid = 1.0
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Package the truth
        # All evidence collected. Signature complete.
        # ─────────────────────────────────────────────────────────────────────
        
        return HarmonicSignature(
            fundamental_freq=fundamental_freq,
            fundamental_amplitude=float(fundamental_amp),
            harmonic_amplitudes=normalized_amps,
            harmonic_phases=harmonic_phases,
            thd=float(thd),
            h2_h1_ratio=float(h2_h1),
            h3_h2_ratio=float(h3_h2),
            odd_even_ratio=float(odd_even_ratio),
            phase_coherence=float(phase_coherence),
            spectral_slope=spectral_slope,
            spectral_centroid=spectral_centroid
        )
    
    def analyze_multi_frequency(self, 
                               captures: Dict[str, 'UltrasoundCapture'],
                               roi_mask: Optional[np.ndarray] = None) -> UltrasoundHarmonicFeatures:
        """
        Analyze harmonics across multiple frequency captures.
        
        One frequency can be fooled.
        Multiple frequencies triangulate the truth.
        
        Args:
            captures: Dictionary of UltrasoundCapture objects - the full spectrum
            roi_mask: Optional region of interest mask
            
        Returns:
            Complete UltrasoundHarmonicFeatures - the comprehensive dossier
        """
        from .ultrasound_capture import UltrasoundCapture
        
        # Categorize signatures by frequency range
        surface_harmonics = {}    # kHz range - surface only
        clinical_harmonics = {}   # Low MHz - dermal penetration
        highfreq_harmonics = {}   # High MHz - cellular detail
        all_signatures = {}
        
        # ─────────────────────────────────────────────────────────────────────
        # Analyze each frequency capture
        # Each is a different question to the tissue
        # ─────────────────────────────────────────────────────────────────────
        
        for name, capture in captures.items():
            # Extract the harmonic signature at this frequency
            signature = self.analyze_capture(
                capture.rf_data,
                capture.center_frequency,
                capture.sampling_rate,
                roi_mask
            )
            
            all_signatures[capture.center_frequency] = signature
            
            # Categorize by frequency range
            # Each range probes a different depth
            freq_mhz = capture.center_frequency / 1e6
            
            if freq_mhz < 1:
                # Surface acoustic (kHz range) - texture mapping
                surface_harmonics[name] = signature
            elif freq_mhz < 20:
                # Clinical ultrasound - tissue penetration
                clinical_harmonics[name] = signature
            else:
                # High frequency - near-cellular resolution
                highfreq_harmonics[name] = signature
        
        # ─────────────────────────────────────────────────────────────────────
        # Build the multi-frequency profile
        # The synthesis of all frequencies
        # ─────────────────────────────────────────────────────────────────────
        
        profile = self._build_frequency_profile(all_signatures)
        
        # ─────────────────────────────────────────────────────────────────────
        # Generate spatial harmonic maps (if 2D data available)
        # Where in space is the tissue abnormal?
        # ─────────────────────────────────────────────────────────────────────
        
        harmonic_image = None
        thd_map = None
        
        # Find a clinical capture with 2D data for spatial analysis
        for name, capture in captures.items():
            if capture.rf_data.ndim == 2 and capture.rf_data.shape[1] > 10:
                # Generate 2nd harmonic image - shows tissue interfaces
                harmonic_image = self._generate_harmonic_image(
                    capture.rf_data, capture.center_frequency, capture.sampling_rate
                )
                # Generate THD map - shows where distortion is highest
                thd_map = self._generate_thd_map(
                    capture.rf_data, capture.center_frequency, capture.sampling_rate
                )
                break
        
        # ─────────────────────────────────────────────────────────────────────
        # Calculate SCC-specific markers
        # The final verdict
        # ─────────────────────────────────────────────────────────────────────
        
        # The SCC score - how cancer-like is this harmonic pattern?
        scc_score = self._calculate_scc_score(all_signatures)
        
        # Depth analysis - how deep does the abnormality extend?
        depth_penetration = self._analyze_depth_penetration(all_signatures)
        
        # Structural irregularity - how chaotic is the tissue?
        structural_irreg = self._calculate_structural_irregularity(all_signatures)
        
        return UltrasoundHarmonicFeatures(
            surface_harmonics=surface_harmonics,
            clinical_harmonics=clinical_harmonics,
            highfreq_harmonics=highfreq_harmonics,
            profile=profile,
            harmonic_image=harmonic_image,
            thd_map=thd_map,
            scc_harmonic_score=scc_score,
            depth_penetration=depth_penetration,
            structural_irregularity=structural_irreg
        )
    
    def _build_frequency_profile(self, 
                                signatures: Dict[float, HarmonicSignature]) -> MultiFrequencyHarmonicProfile:
        """
        Build cross-frequency harmonic profile.
        
        The forest, not just the trees.
        """
        if not signatures:
            # No data, no profile
            return MultiFrequencyHarmonicProfile(
                signatures={},
                harmonic_consistency=0.0,
                frequency_dependence={},
                mean_thd=0.0,
                max_thd=0.0,
                thd_variance=0.0
            )
        
        # THD statistics across all frequencies
        thd_values = [sig.thd for sig in signatures.values()]
        mean_thd = float(np.mean(thd_values))
        max_thd = float(np.max(thd_values))
        thd_variance = float(np.var(thd_values))  # High variance is suspicious
        
        # Harmonic consistency across frequencies
        # How stable are harmonic ratios?
        h2_h1_values = [sig.h2_h1_ratio for sig in signatures.values()]
        harmonic_consistency = 1.0 - min(1.0, np.std(h2_h1_values) / (np.mean(h2_h1_values) + 1e-10))
        
        # Frequency dependence of each harmonic
        # Does a particular harmonic change with frequency?
        frequency_dependence = {}
        freqs = sorted(signatures.keys())
        
        for order in range(2, self.max_harmonic_order + 1):
            amps = []
            for freq in freqs:
                sig = signatures[freq]
                amp = sig.harmonic_amplitudes.get(order, 0.0)
                amps.append(amp)
            
            if len(amps) > 2:
                # Fit linear trend in log-frequency space
                log_freqs = np.log(np.array(freqs) + 1)
                slope, _ = np.polyfit(log_freqs, amps, 1)
                frequency_dependence[order] = float(slope)
        
        return MultiFrequencyHarmonicProfile(
            signatures=signatures,
            harmonic_consistency=float(harmonic_consistency),
            frequency_dependence=frequency_dependence,
            mean_thd=mean_thd,
            max_thd=max_thd,
            thd_variance=thd_variance
        )
    
    def _generate_harmonic_image(self,
                                rf_data: np.ndarray,
                                fundamental_freq: float,
                                sampling_rate: float,
                                harmonic_order: int = 2) -> np.ndarray:
        """
        Generate spatial image from specific harmonic content.
        
        Second harmonic imaging is particularly useful for tissue characterization.
        The second harmonic reveals what the fundamental cannot.
        
        "Where the second harmonic rises, there the tissue is transformed."
        """
        n_samples, n_lines = rf_data.shape
        harmonic_image = np.zeros((n_samples, n_lines))
        
        # Design bandpass filter around target harmonic
        target_freq = harmonic_order * fundamental_freq
        bandwidth = fundamental_freq * 0.5
        
        nyq = sampling_rate / 2
        low = max(0.01, (target_freq - bandwidth) / nyq)
        high = min(0.99, (target_freq + bandwidth) / nyq)
        
        if low < high:
            try:
                # Butterworth bandpass - smooth, reliable
                b, a = butter(4, [low, high], btype='band')
                
                # Filter each scan line and extract envelope
                for line_idx in range(n_lines):
                    filtered = filtfilt(b, a, rf_data[:, line_idx])
                    harmonic_image[:, line_idx] = np.abs(hilbert(filtered))
            except:
                # If filter design fails, fall back to simple envelope
                # Not ideal, but better than nothing
                harmonic_image = np.abs(hilbert(rf_data, axis=0))
        else:
            harmonic_image = np.abs(hilbert(rf_data, axis=0))
        
        return harmonic_image
    
    def _generate_thd_map(self,
                         rf_data: np.ndarray,
                         fundamental_freq: float,
                         sampling_rate: float,
                         window_size: int = 64) -> np.ndarray:
        """
        Generate spatial map of Total Harmonic Distortion.
        
        High THD regions may indicate abnormal tissue.
        This map shows WHERE the tissue is misbehaving.
        
        "The THD map is a heat map of acoustic sin."
        """
        n_samples, n_lines = rf_data.shape
        
        # Output at reduced resolution (one THD value per window)
        out_samples = n_samples // window_size
        thd_map = np.zeros((out_samples, n_lines))
        
        # Calculate THD for each spatial window
        for line_idx in range(n_lines):
            for win_idx in range(out_samples):
                start = win_idx * window_size
                end = min(start + window_size, n_samples)
                
                segment = rf_data[start:end, line_idx]
                
                if len(segment) < window_size // 2:
                    continue
                
                # Quick THD calculation for this window
                segment_centered = segment - np.mean(segment)
                
                fft_seg = np.fft.fft(segment_centered)
                freqs = np.fft.fftfreq(len(segment_centered), 1/sampling_rate)
                
                pos_mask = freqs > 0
                magnitude = np.abs(fft_seg[pos_mask])
                freqs_pos = freqs[pos_mask]
                
                # Find fundamental amplitude
                fund_idx = np.argmin(np.abs(freqs_pos - fundamental_freq))
                fund_amp = magnitude[fund_idx] + 1e-10
                
                # Sum harmonic power (2nd through 5th for speed)
                harmonic_power = 0
                for order in range(2, 6):
                    harm_freq = order * fundamental_freq
                    if harm_freq < sampling_rate / 2:
                        harm_idx = np.argmin(np.abs(freqs_pos - harm_freq))
                        harmonic_power += magnitude[harm_idx] ** 2
                
                thd = np.sqrt(harmonic_power) / fund_amp
                thd_map[win_idx, line_idx] = thd
        
        return thd_map
    
    def _calculate_scc_score(self, signatures: Dict[float, HarmonicSignature]) -> float:
        """
        Calculate SCC-specific score based on harmonic patterns.
        
        THIS IS THE VERDICT.
        
        SCC characteristics we're looking for:
        - Elevated THD (increased tissue non-linearity)
        - Higher H2/H1 ratio (changed elastic properties)
        - Irregular harmonic ratios (heterogeneous structure)
        - Frequency-dependent changes (depth-varying properties)
        
        Returns a score from 0 (normal) to 1 (highly suspicious for SCC).
        """
        if not signatures:
            return 0.0
        
        score = 0.0
        weights = []
        
        for freq, sig in signatures.items():
            # ─────────────────────────────────────────────────────────────────
            # THD elevation - the primary marker
            # Cancer tissue distorts more than healthy tissue
            # ─────────────────────────────────────────────────────────────────
            thd_elevation = (sig.thd - self.normal_skin_thd) / self.normal_skin_thd
            thd_score = min(1.0, max(0.0, thd_elevation))
            
            # ─────────────────────────────────────────────────────────────────
            # H2/H1 elevation - the second harmonic betrayal
            # Changed elastic properties generate more H2
            # ─────────────────────────────────────────────────────────────────
            h2_elevation = (sig.h2_h1_ratio - self.normal_skin_h2_h1) / self.normal_skin_h2_h1
            h2_score = min(1.0, max(0.0, h2_elevation))
            
            # ─────────────────────────────────────────────────────────────────
            # Odd/even harmonic imbalance
            # Cancer often shifts this balance
            # ─────────────────────────────────────────────────────────────────
            odd_even_score = min(1.0, max(0.0, sig.odd_even_ratio - 1.0))
            
            # ─────────────────────────────────────────────────────────────────
            # Phase coherence reduction - structural chaos
            # Irregular structure → random phases
            # ─────────────────────────────────────────────────────────────────
            phase_score = 1.0 - sig.phase_coherence
            
            # Combined score for this frequency
            freq_score = (thd_score * 0.3 + 
                         h2_score * 0.3 + 
                         odd_even_score * 0.2 + 
                         phase_score * 0.2)
            
            # Weight by frequency
            # Higher frequencies are more specific to superficial lesions
            weight = 1.0 + np.log10(freq / 1e6 + 1)
            
            score += freq_score * weight
            weights.append(weight)
        
        # Normalize by total weight
        return float(score / (sum(weights) + 1e-10))
    
    def _analyze_depth_penetration(self, signatures: Dict[float, HarmonicSignature]) -> float:
        """
        Analyze how deep the abnormality extends based on frequency response.
        
        Lower frequencies penetrate deeper.
        If low frequencies show abnormality = deep lesion.
        If only high frequencies show abnormality = superficial lesion.
        
        "The depth of penetration reveals the depth of the disease."
        """
        if len(signatures) < 3:
            return 0.0
        
        # Sort by frequency
        sorted_freqs = sorted(signatures.keys())
        
        # Compare low vs high frequency harmonic patterns
        low_freqs = sorted_freqs[:len(sorted_freqs)//3]
        high_freqs = sorted_freqs[-len(sorted_freqs)//3:]
        
        low_thd = np.mean([signatures[f].thd for f in low_freqs])
        high_thd = np.mean([signatures[f].thd for f in high_freqs])
        
        # If low frequency shows similar THD as high frequency,
        # the lesion likely extends deeper
        if high_thd > 0:
            depth_indicator = low_thd / high_thd
            return float(min(1.0, depth_indicator))
        
        return 0.0
    
    def _calculate_structural_irregularity(self, 
                                          signatures: Dict[float, HarmonicSignature]) -> float:
        """
        Calculate structural irregularity from harmonic variance.
        
        Irregular structure → inconsistent harmonic generation.
        Uniform tissue → consistent harmonics.
        
        "Chaos leaves a harmonic fingerprint."
        """
        if len(signatures) < 2:
            return 0.0
        
        # Variance in spectral slope across frequencies
        slopes = [sig.spectral_slope for sig in signatures.values()]
        slope_variance = np.var(slopes)
        
        # Variance in harmonic ratios
        h2_values = [sig.h2_h1_ratio for sig in signatures.values()]
        h2_variance = np.var(h2_values)
        
        # Variance in phase coherence
        phase_values = [sig.phase_coherence for sig in signatures.values()]
        phase_variance = np.var(phase_values)
        
        # Combined irregularity score
        irregularity = (slope_variance * 0.3 + 
                       h2_variance * 0.4 + 
                       phase_variance * 0.3)
        
        return float(min(1.0, irregularity * 10))  # Scale to 0-1
    
    def generate_feature_vector(self, features: UltrasoundHarmonicFeatures) -> np.ndarray:
        """
        Generate flat feature vector for ML input.
        
        The machine requires numbers.
        We give it the essence of our analysis.
        
        Returns:
            1D numpy array with all harmonic features
        """
        vector = []
        
        # Profile-level features
        vector.extend([
            features.profile.mean_thd,
            features.profile.max_thd,
            features.profile.thd_variance,
            features.profile.harmonic_consistency
        ])
        
        # Frequency dependence for each harmonic
        for order in range(2, self.max_harmonic_order + 1):
            dep = features.profile.frequency_dependence.get(order, 0.0)
            vector.append(dep)
        
        # Aggregate per-category statistics
        for category_sigs in [features.surface_harmonics, 
                              features.clinical_harmonics,
                              features.highfreq_harmonics]:
            if category_sigs:
                thds = [s.thd for s in category_sigs.values()]
                h2s = [s.h2_h1_ratio for s in category_sigs.values()]
                vector.extend([
                    np.mean(thds),
                    np.std(thds),
                    np.mean(h2s),
                    np.std(h2s)
                ])
            else:
                vector.extend([0.0, 0.0, 0.0, 0.0])
        
        # SCC-specific features - the crown jewels
        vector.extend([
            features.scc_harmonic_score,
            features.depth_penetration,
            features.structural_irregularity
        ])
        
        # THD map statistics if available
        if features.thd_map is not None:
            vector.extend([
                np.mean(features.thd_map),
                np.std(features.thd_map),
                np.max(features.thd_map),
                np.percentile(features.thd_map, 95)
            ])
        else:
            vector.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(vector, dtype=np.float32)
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get names for the feature vector - for interpretability"""
        names = [
            'harmonic_mean_thd', 'harmonic_max_thd', 
            'harmonic_thd_variance', 'harmonic_consistency'
        ]
        
        for order in range(2, 9):
            names.append(f'freq_dependence_h{order}')
        
        for category in ['surface', 'clinical', 'highfreq']:
            names.extend([
                f'{category}_mean_thd',
                f'{category}_std_thd',
                f'{category}_mean_h2',
                f'{category}_std_h2'
            ])
        
        names.extend([
            'scc_harmonic_score',
            'depth_penetration',
            'structural_irregularity',
            'thd_map_mean',
            'thd_map_std',
            'thd_map_max',
            'thd_map_p95'
        ])
        
        return names
    
    def explain_harmonic_findings(self, features: UltrasoundHarmonicFeatures) -> str:
        """
        Generate human-readable explanation of harmonic findings.
        
        For clinical interpretability.
        The machine knows. Now the human must understand.
        
        "Data without explanation is merely noise.
         Explanation without data is merely opinion."
        """
        report = []
        report.append("=" * 60)
        report.append("ULTRASOUND HARMONIC ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Overall assessment - the headline
        scc_score = features.scc_harmonic_score
        if scc_score > 0.7:
            report.append(f"\n⚠️  HIGH SUSPICION: SCC Score = {scc_score:.2f}")
        elif scc_score > 0.4:
            report.append(f"\n⚡ MODERATE SUSPICION: SCC Score = {scc_score:.2f}")
        else:
            report.append(f"\n✓ LOW SUSPICION: SCC Score = {scc_score:.2f}")
        
        # Profile summary - the numbers
        report.append(f"\n📊 Harmonic Profile:")
        report.append(f"   Mean THD: {features.profile.mean_thd:.3f} (normal: <{self.normal_skin_thd})")
        report.append(f"   Max THD: {features.profile.max_thd:.3f}")
        report.append(f"   Harmonic Consistency: {features.profile.harmonic_consistency:.2f}")
        
        # Depth analysis - how deep?
        report.append(f"\n📏 Depth Analysis:")
        report.append(f"   Penetration Index: {features.depth_penetration:.2f}")
        if features.depth_penetration > 0.7:
            report.append("   → Suggests deep tissue involvement")
        elif features.depth_penetration > 0.4:
            report.append("   → Suggests dermal involvement")
        else:
            report.append("   → Suggests superficial involvement")
        
        # Structural irregularity - how chaotic?
        report.append(f"\n🔍 Structural Analysis:")
        report.append(f"   Irregularity Score: {features.structural_irregularity:.2f}")
        if features.structural_irregularity > 0.6:
            report.append("   → Highly irregular structure (suspicious)")
        elif features.structural_irregularity > 0.3:
            report.append("   → Moderately irregular structure")
        else:
            report.append("   → Regular structure")
        
        # Frequency-specific findings - the details
        report.append(f"\n📡 Frequency Analysis:")
        
        if features.surface_harmonics:
            thd = np.mean([s.thd for s in features.surface_harmonics.values()])
            report.append(f"   Surface (40-200 kHz): THD = {thd:.3f}")
        
        if features.clinical_harmonics:
            thd = np.mean([s.thd for s in features.clinical_harmonics.values()])
            report.append(f"   Clinical (5-15 MHz): THD = {thd:.3f}")
        
        if features.highfreq_harmonics:
            thd = np.mean([s.thd for s in features.highfreq_harmonics.values()])
            report.append(f"   High-Freq (20-50 MHz): THD = {thd:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# ═══════════════════════════════════════════════════════════════════════════════
# END OF MODULE
# 
# "The tissue speaks in harmonics.
#  We have learned to listen.
#  Now let us act on what we hear."
# ═══════════════════════════════════════════════════════════════════════════════
