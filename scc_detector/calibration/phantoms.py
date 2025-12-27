"""
Tissue Phantoms for System Calibration and Validation

Phantoms simulate tissue properties for:
1. Visual system calibration (color targets)
2. Thermal calibration (known temperature references)
3. Acoustic calibration (known acoustic properties)
4. System validation (known lesion characteristics)

Materials:
- Agar-based phantoms for ultrasound
- Silicone phantoms for optical properties
- Graphite/intralipid for scattering
- Microbubbles for harmonic response
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PhantomProperties:
    """Physical properties of a tissue phantom"""
    # Optical properties
    absorption_coefficient: float  # mm^-1
    scattering_coefficient: float  # mm^-1
    anisotropy_factor: float  # g
    refractive_index: float
    
    # Acoustic properties
    speed_of_sound: float  # m/s
    acoustic_attenuation: float  # dB/cm/MHz
    density: float  # kg/m^3
    nonlinearity_parameter: float  # B/A
    
    # Thermal properties
    thermal_conductivity: float  # W/m·K
    specific_heat: float  # J/kg·K
    thermal_diffusivity: float  # mm^2/s
    
    # Geometric properties
    thickness_mm: float
    diameter_mm: float
    
    # Phantom type
    phantom_type: str  # "skin", "lesion", "calibration"


@dataclass
class CalibrationResult:
    """Results from calibration procedure"""
    timestamp: datetime
    
    # Visual calibration
    color_error: float  # ΔE in LAB
    white_balance_correction: Tuple[float, float, float]
    exposure_correction: float
    
    # Thermal calibration
    temperature_offset: float  # °C
    temperature_gain: float
    ambient_temp: float
    
    # Acoustic calibration
    speed_of_sound_measured: float
    attenuation_measured: float
    harmonic_response_measured: Dict[int, float]
    
    # Size calibration
    mm_per_pixel: float
    depth_calibration: float  # mm
    
    # Overall status
    calibration_valid: bool
    notes: str


class TissuePhantom:
    """
    Tissue-mimicking phantom for system calibration.
    
    Recipe for common phantoms included.
    """
    
    # Reference tissue properties
    TISSUE_PROPERTIES = {
        'skin_epidermis': PhantomProperties(
            absorption_coefficient=0.5,
            scattering_coefficient=40.0,
            anisotropy_factor=0.8,
            refractive_index=1.4,
            speed_of_sound=1540,
            acoustic_attenuation=1.5,
            density=1100,
            nonlinearity_parameter=6.5,
            thermal_conductivity=0.3,
            specific_heat=3500,
            thermal_diffusivity=0.1,
            thickness_mm=0.1,
            diameter_mm=50,
            phantom_type="skin"
        ),
        'skin_dermis': PhantomProperties(
            absorption_coefficient=0.2,
            scattering_coefficient=20.0,
            anisotropy_factor=0.75,
            refractive_index=1.38,
            speed_of_sound=1580,
            acoustic_attenuation=1.0,
            density=1050,
            nonlinearity_parameter=6.0,
            thermal_conductivity=0.5,
            specific_heat=3800,
            thermal_diffusivity=0.12,
            thickness_mm=2.0,
            diameter_mm=50,
            phantom_type="skin"
        ),
        'scc_lesion': PhantomProperties(
            absorption_coefficient=0.8,  # Higher due to increased vascularity
            scattering_coefficient=60.0,  # More irregular structure
            anisotropy_factor=0.7,
            refractive_index=1.42,
            speed_of_sound=1520,  # Slightly lower
            acoustic_attenuation=2.0,  # Higher attenuation
            density=1080,
            nonlinearity_parameter=8.0,  # Higher nonlinearity
            thermal_conductivity=0.4,
            specific_heat=3600,
            thermal_diffusivity=0.11,
            thickness_mm=5.0,
            diameter_mm=10,
            phantom_type="lesion"
        ),
        'benign_lesion': PhantomProperties(
            absorption_coefficient=0.4,
            scattering_coefficient=35.0,
            anisotropy_factor=0.78,
            refractive_index=1.39,
            speed_of_sound=1550,
            acoustic_attenuation=1.2,
            density=1060,
            nonlinearity_parameter=6.2,
            thermal_conductivity=0.35,
            specific_heat=3600,
            thermal_diffusivity=0.1,
            thickness_mm=3.0,
            diameter_mm=8,
            phantom_type="lesion"
        )
    }
    
    def __init__(self, phantom_type: str = "skin_dermis"):
        """
        Initialize phantom with given type.
        
        Args:
            phantom_type: Type from TISSUE_PROPERTIES
        """
        if phantom_type in self.TISSUE_PROPERTIES:
            self.properties = self.TISSUE_PROPERTIES[phantom_type]
        else:
            print(f"⚠️  Unknown phantom type '{phantom_type}', using skin_dermis")
            self.properties = self.TISSUE_PROPERTIES['skin_dermis']
        
        self.phantom_type = phantom_type
    
    @classmethod
    def get_recipe(cls, phantom_type: str) -> str:
        """
        Get recipe for creating a tissue phantom.
        
        Args:
            phantom_type: Type of phantom to create
            
        Returns:
            Recipe instructions
        """
        recipes = {
            'skin_agar': """
AGAR-BASED SKIN PHANTOM RECIPE

Materials needed:
- 500 mL distilled water
- 15 g agar powder (3% w/v)
- 10 g intralipid 20% solution (optical scattering)
- 2 mL India ink (optical absorption)
- 5 g graphite powder (ultrasound scattering)

Procedure:
1. Heat 400 mL water to 90°C
2. Slowly add agar while stirring constantly
3. Once dissolved, reduce heat to 60°C
4. Add intralipid and mix thoroughly
5. Add graphite powder and mix
6. Add India ink drop by drop to achieve desired absorption
7. Pour into mold and let cool to room temperature
8. Refrigerate for 2 hours before use

Storage: Refrigerate, use within 1 week
Shelf life can be extended by adding 0.1% sodium azide
""",
            'silicone_skin': """
SILICONE-BASED SKIN PHANTOM RECIPE

Materials needed:
- 100 g platinum-cure silicone (shore 10A)
- 3 g titanium dioxide (scattering)
- 0.5 g iron oxide pigment (absorption)
- 10 g silicone thinner
- 5 g glass microspheres (optional, for ultrasound)

Procedure:
1. Mix Part A silicone with pigments
2. Add thinner and mix thoroughly
3. Add glass microspheres if needed
4. Add Part B catalyst
5. Degas in vacuum chamber for 10 minutes
6. Pour into mold
7. Cure at room temperature for 24 hours
   Or cure at 60°C for 4 hours

Storage: Stable at room temperature for months
""",
            'lesion_inclusion': """
LESION INCLUSION PHANTOM

Base phantom recipe (as above) plus:

For SCC-mimicking lesion:
- Increase graphite to 10 g (higher scattering)
- Add 0.1 mL red food dye (vascularity)
- Add 1 g microbubbles (harmonic response)

Procedure:
1. Make base phantom but pour only half
2. Let partially set (30 min)
3. Place spherical lesion inclusion (5-10 mm diameter)
4. Pour remaining phantom material
5. Let set completely

Lesion inclusion preparation:
- Same recipe as base but with modified scattering
- Mold in silicone sphere molds
"""
        }
        
        return recipes.get(phantom_type, recipes['skin_agar'])
    
    def simulate_visual_response(self, illumination: str = "white") -> np.ndarray:
        """
        Simulate visual appearance of phantom.
        
        Args:
            illumination: "white", "uv", or wavelength in nm
            
        Returns:
            Simulated RGB image
        """
        size = int(self.properties.diameter_mm * 10)
        image = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Base color based on optical properties
        absorption = self.properties.absorption_coefficient
        scattering = self.properties.scattering_coefficient
        
        # Higher absorption = darker, higher scattering = more diffuse
        if self.properties.phantom_type == "lesion":
            # Lesion - reddish-brown
            base_color = np.array([140, 100, 80])  # RGB
        else:
            # Normal skin
            base_color = np.array([220, 190, 170])  # RGB
        
        # Apply optical effects
        color = base_color * np.exp(-absorption * 0.5)
        color = np.clip(color, 0, 255).astype(np.uint8)
        
        # Fill image
        center = (size // 2, size // 2)
        radius = size // 2 - 5
        
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist < radius:
                    # Add some texture noise
                    noise = np.random.normal(0, 5, 3)
                    pixel = np.clip(color + noise, 0, 255)
                    image[y, x] = pixel.astype(np.uint8)
        
        return image
    
    def simulate_thermal_response(self, ambient_temp: float = 25.0) -> np.ndarray:
        """
        Simulate thermal image of phantom.
        
        Args:
            ambient_temp: Ambient temperature in Celsius
            
        Returns:
            Temperature map in Celsius
        """
        size = int(self.properties.diameter_mm * 2)
        thermal = np.ones((size, size)) * ambient_temp
        
        center = (size // 2, size // 2)
        radius = size // 2 - 2
        
        # Phantom at higher temperature than ambient
        phantom_temp = ambient_temp + 5.0
        
        for y in range(size):
            for x in range(size):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist < radius:
                    # Temperature based on thermal properties
                    conductivity = self.properties.thermal_conductivity
                    
                    if self.properties.phantom_type == "lesion":
                        # Lesion slightly warmer
                        thermal[y, x] = phantom_temp + 1.0 + np.random.normal(0, 0.1)
                    else:
                        thermal[y, x] = phantom_temp + np.random.normal(0, 0.1)
        
        return thermal
    
    def simulate_acoustic_response(self, 
                                   frequency: float = 10e6,
                                   sampling_rate: float = 40e6) -> np.ndarray:
        """
        Simulate ultrasound RF signal from phantom.
        
        Args:
            frequency: Transmit frequency in Hz
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Simulated RF signal
        """
        # Calculate time array for depth
        c = self.properties.speed_of_sound
        max_depth = self.properties.thickness_mm / 1000  # Convert to meters
        max_time = 2 * max_depth / c
        
        samples = int(max_time * sampling_rate)
        t = np.arange(samples) / sampling_rate
        
        # Initialize signal
        signal = np.zeros(samples)
        
        # Add echo at phantom surface
        surface_time = self.properties.thickness_mm / 1000 / c * 2
        surface_idx = int(surface_time * sampling_rate)
        
        if surface_idx < samples:
            # Echo with depth-dependent attenuation
            depth_mm = c * t / 2 * 1000
            attenuation = 10 ** (-self.properties.acoustic_attenuation * depth_mm / 10 * frequency / 1e6)
            
            # Fundamental
            signal += attenuation * np.sin(2 * np.pi * frequency * t)
            
            # Add harmonics based on nonlinearity parameter
            B_A = self.properties.nonlinearity_parameter
            
            for harmonic in range(2, 6):
                if harmonic * frequency < sampling_rate / 2:
                    # Harmonic amplitude proportional to (B/A)^(n-1)
                    harmonic_amp = (B_A / 6) ** (harmonic - 1) * 0.3
                    signal += attenuation * harmonic_amp * np.sin(2 * np.pi * harmonic * frequency * t)
            
            # Add speckle noise
            signal += np.random.normal(0, 0.1, samples)
        
        return signal
    
    def get_expected_harmonic_ratios(self) -> Dict[int, float]:
        """
        Get expected harmonic ratios for this phantom.
        
        Returns:
            Dictionary of harmonic order to expected relative amplitude
        """
        B_A = self.properties.nonlinearity_parameter
        
        ratios = {}
        for harmonic in range(1, 9):
            # Simplified model: amplitude ~ (B/A)^(n-1) / n!
            import math
            ratio = (B_A / 6) ** (harmonic - 1) / math.factorial(harmonic)
            ratios[harmonic] = ratio / ratios.get(1, 1.0)
        
        # Normalize to fundamental
        fund = ratios[1]
        ratios = {k: v / fund for k, v in ratios.items()}
        
        return ratios


class CalibrationSystem:
    """
    Complete calibration system for all modalities.
    
    Calibrates:
    - Visual: color accuracy, white balance, exposure
    - Thermal: temperature accuracy
    - Acoustic: speed of sound, attenuation, harmonics
    - Dimensional: mm per pixel
    """
    
    def __init__(self):
        """Initialize calibration system"""
        self.calibration_results: List[CalibrationResult] = []
        self.current_calibration: Optional[CalibrationResult] = None
        
        # Reference values
        self.color_reference_lab = np.array([50, 0, 0])  # Neutral gray
        self.temperature_reference = 37.0  # Body temperature
    
    def run_full_calibration(self,
                            visual_capture: Optional[np.ndarray] = None,
                            thermal_capture: Optional[np.ndarray] = None,
                            acoustic_signal: Optional[np.ndarray] = None,
                            calibration_target_mm: float = 20.0) -> CalibrationResult:
        """
        Run complete system calibration.
        
        Args:
            visual_capture: Image of color calibration target
            thermal_capture: Thermal image of temperature reference
            acoustic_signal: Ultrasound signal from calibration phantom
            calibration_target_mm: Known size of calibration target
            
        Returns:
            CalibrationResult with all corrections
        """
        print("\n" + "=" * 60)
        print("    SYSTEM CALIBRATION")
        print("=" * 60)
        
        result = CalibrationResult(
            timestamp=datetime.now(),
            color_error=0.0,
            white_balance_correction=(1.0, 1.0, 1.0),
            exposure_correction=1.0,
            temperature_offset=0.0,
            temperature_gain=1.0,
            ambient_temp=25.0,
            speed_of_sound_measured=1540.0,
            attenuation_measured=1.0,
            harmonic_response_measured={},
            mm_per_pixel=0.1,
            depth_calibration=1.0,
            calibration_valid=True,
            notes=""
        )
        
        notes = []
        
        # Visual calibration
        if visual_capture is not None:
            print("\n[1/4] Visual calibration...")
            color_error, wb_corr, exp_corr = self._calibrate_visual(visual_capture)
            result.color_error = color_error
            result.white_balance_correction = wb_corr
            result.exposure_correction = exp_corr
            
            if color_error > 3.0:
                notes.append(f"High color error ({color_error:.1f} ΔE)")
            print(f"      Color error: {color_error:.2f} ΔE")
        else:
            print("\n[1/4] Visual calibration skipped (no capture)")
        
        # Thermal calibration
        if thermal_capture is not None:
            print("\n[2/4] Thermal calibration...")
            temp_offset, temp_gain, ambient = self._calibrate_thermal(thermal_capture)
            result.temperature_offset = temp_offset
            result.temperature_gain = temp_gain
            result.ambient_temp = ambient
            print(f"      Temperature offset: {temp_offset:.2f}°C")
        else:
            print("\n[2/4] Thermal calibration skipped (no capture)")
        
        # Acoustic calibration
        if acoustic_signal is not None:
            print("\n[3/4] Acoustic calibration...")
            sos, atten, harmonics = self._calibrate_acoustic(acoustic_signal)
            result.speed_of_sound_measured = sos
            result.attenuation_measured = atten
            result.harmonic_response_measured = harmonics
            print(f"      Speed of sound: {sos:.0f} m/s")
            print(f"      Attenuation: {atten:.2f} dB/cm/MHz")
        else:
            print("\n[3/4] Acoustic calibration skipped (no signal)")
        
        # Dimensional calibration
        if visual_capture is not None:
            print("\n[4/4] Dimensional calibration...")
            mm_per_pixel = self._calibrate_dimensions(visual_capture, calibration_target_mm)
            result.mm_per_pixel = mm_per_pixel
            print(f"      Scale: {mm_per_pixel:.4f} mm/pixel")
        else:
            print("\n[4/4] Dimensional calibration skipped")
        
        result.notes = "; ".join(notes) if notes else "Calibration successful"
        
        # Validate overall calibration
        if result.color_error > 5.0 or abs(result.temperature_offset) > 2.0:
            result.calibration_valid = False
            result.notes += " - CALIBRATION MARGINAL"
        
        self.current_calibration = result
        self.calibration_results.append(result)
        
        print("\n" + "=" * 60)
        print(f"✓ Calibration complete - Valid: {result.calibration_valid}")
        print("=" * 60)
        
        return result
    
    def _calibrate_visual(self, 
                         image: np.ndarray) -> Tuple[float, Tuple[float, float, float], float]:
        """Calibrate visual system using color target"""
        import cv2
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Find gray patch (assuming it's in center)
        h, w = image.shape[:2]
        center_region = image[h//3:2*h//3, w//3:2*w//3]
        
        # Measure color
        measured_lab = np.array([
            np.mean(cv2.cvtColor(center_region, cv2.COLOR_BGR2LAB)[:,:,i])
            for i in range(3)
        ])
        
        # Calculate color error (ΔE)
        delta_e = np.sqrt(np.sum((measured_lab - self.color_reference_lab) ** 2))
        
        # Calculate white balance correction
        bgr_mean = np.mean(center_region, axis=(0, 1))
        gray_val = np.mean(bgr_mean)
        wb_correction = tuple(gray_val / (c + 1e-10) for c in bgr_mean)
        
        # Calculate exposure correction
        target_brightness = 128
        exp_correction = target_brightness / (np.mean(l) + 1e-10)
        
        return float(delta_e), wb_correction, float(exp_correction)
    
    def _calibrate_thermal(self, 
                          thermal: np.ndarray) -> Tuple[float, float, float]:
        """Calibrate thermal system"""
        # Assume thermal image in Celsius
        measured_temp = np.mean(thermal)
        ambient_temp = np.percentile(thermal, 10)  # Coolest region
        
        # Reference is body temperature phantom
        temp_offset = self.temperature_reference - measured_temp
        
        # Gain (simplified)
        temp_gain = 1.0
        
        return float(temp_offset), float(temp_gain), float(ambient_temp)
    
    def _calibrate_acoustic(self, 
                           signal: np.ndarray,
                           sampling_rate: float = 40e6,
                           phantom_thickness: float = 20e-3) -> Tuple[float, float, Dict]:
        """Calibrate acoustic system"""
        # Find echo (peak detection)
        from scipy.signal import hilbert, find_peaks
        
        envelope = np.abs(hilbert(signal))
        peaks, _ = find_peaks(envelope, height=np.max(envelope) * 0.3)
        
        if len(peaks) >= 2:
            # Time between first two peaks (round trip through phantom)
            round_trip_samples = peaks[1] - peaks[0]
            round_trip_time = round_trip_samples / sampling_rate
            
            # Speed of sound
            speed_of_sound = 2 * phantom_thickness / round_trip_time
        else:
            speed_of_sound = 1540.0  # Default
        
        # Attenuation (amplitude ratio between echoes)
        if len(peaks) >= 2:
            amp_ratio = envelope[peaks[1]] / (envelope[peaks[0]] + 1e-10)
            # Attenuation in dB
            attenuation_db = -20 * np.log10(amp_ratio + 1e-10)
            attenuation = attenuation_db / (phantom_thickness * 1000 * 10)  # per cm per MHz
        else:
            attenuation = 1.0
        
        # Harmonic response
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        
        harmonics = {}
        fundamental_freq = 10e6  # Assumed
        
        for order in range(1, 6):
            target_freq = order * fundamental_freq
            if target_freq < sampling_rate / 2:
                idx = np.argmin(np.abs(freqs - target_freq))
                harmonics[order] = float(np.abs(fft[idx]))
        
        # Normalize to fundamental
        if 1 in harmonics and harmonics[1] > 0:
            harmonics = {k: v / harmonics[1] for k, v in harmonics.items()}
        
        return float(speed_of_sound), float(attenuation), harmonics
    
    def _calibrate_dimensions(self, 
                             image: np.ndarray,
                             target_size_mm: float) -> float:
        """Calibrate dimensional measurements"""
        import cv2
        
        # Detect calibration target (assuming circular)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=200
        )
        
        if circles is not None:
            # Use largest circle
            circles = np.round(circles[0, :]).astype("int")
            largest = max(circles, key=lambda c: c[2])
            diameter_pixels = largest[2] * 2
            
            mm_per_pixel = target_size_mm / diameter_pixels
        else:
            # Default calibration
            mm_per_pixel = 0.1
        
        return float(mm_per_pixel)
    
    def apply_calibration(self,
                         image: Optional[np.ndarray] = None,
                         thermal: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply current calibration to captures.
        
        Args:
            image: Visual image to calibrate
            thermal: Thermal image to calibrate
            
        Returns:
            (calibrated_image, calibrated_thermal)
        """
        if self.current_calibration is None:
            print("⚠️  No calibration available")
            return image, thermal
        
        cal = self.current_calibration
        
        # Calibrate image
        calibrated_image = None
        if image is not None:
            # Apply white balance
            wb = cal.white_balance_correction
            calibrated_image = image.copy().astype(float)
            calibrated_image[:, :, 0] *= wb[0]
            calibrated_image[:, :, 1] *= wb[1]
            calibrated_image[:, :, 2] *= wb[2]
            
            # Apply exposure correction
            calibrated_image *= cal.exposure_correction
            calibrated_image = np.clip(calibrated_image, 0, 255).astype(np.uint8)
        
        # Calibrate thermal
        calibrated_thermal = None
        if thermal is not None:
            calibrated_thermal = thermal * cal.temperature_gain + cal.temperature_offset
        
        return calibrated_image, calibrated_thermal
    
    def get_calibration_report(self) -> str:
        """Generate calibration status report"""
        if self.current_calibration is None:
            return "No calibration performed"
        
        cal = self.current_calibration
        
        report = []
        report.append("=" * 50)
        report.append("CALIBRATION STATUS REPORT")
        report.append("=" * 50)
        report.append(f"Timestamp: {cal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Valid: {'Yes' if cal.calibration_valid else 'NO - RECALIBRATE'}")
        
        report.append("\nVisual Calibration:")
        report.append(f"  Color Error: {cal.color_error:.2f} ΔE")
        report.append(f"  White Balance: R={cal.white_balance_correction[2]:.2f}, "
                     f"G={cal.white_balance_correction[1]:.2f}, "
                     f"B={cal.white_balance_correction[0]:.2f}")
        report.append(f"  Exposure: {cal.exposure_correction:.2f}x")
        
        report.append("\nThermal Calibration:")
        report.append(f"  Temperature Offset: {cal.temperature_offset:+.2f}°C")
        report.append(f"  Ambient Temperature: {cal.ambient_temp:.1f}°C")
        
        report.append("\nAcoustic Calibration:")
        report.append(f"  Speed of Sound: {cal.speed_of_sound_measured:.0f} m/s")
        report.append(f"  Attenuation: {cal.attenuation_measured:.2f} dB/cm/MHz")
        
        report.append("\nDimensional Calibration:")
        report.append(f"  Scale: {cal.mm_per_pixel:.4f} mm/pixel")
        
        if cal.notes:
            report.append(f"\nNotes: {cal.notes}")
        
        report.append("=" * 50)
        
        return "\n".join(report)

