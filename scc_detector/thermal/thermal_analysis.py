"""
Thermal Spectrum Analysis for SCC Detection
============================================

Infrared thermal imaging captures:
- Metabolic heat signatures (cancer cells have higher metabolism)
- Vascular patterns (tumors have increased blood flow)
- Thermal asymmetry (comparing to contralateral tissue)
- Dynamic thermal recovery patterns

    "Cancer burns hotter. Always hotter.
     The uncontrolled proliferation generates heat.
     The anarchic vasculature pools warmth.
     The tumor cannot hide its fever.
     
     Heat is truth."

THE THERMAL INTERROGATION: WHERE METABOLISM CONFESSES.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - The infrared arsenal
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import cv2


@dataclass
class ThermalCapture:
    """Thermal imaging data container"""
    thermal_image: np.ndarray  # Temperature in Celsius
    thermal_raw: np.ndarray  # Raw sensor data
    rgb_aligned: Optional[np.ndarray] = None  # For overlay
    thermal_sequence: List[np.ndarray] = field(default_factory=list)  # For dynamics
    timestamp: datetime = field(default_factory=datetime.now)
    ambient_temp: float = 25.0
    body_location: str = ""
    patient_id: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def get_colormap_image(self, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Convert thermal data to colormap visualization"""
        # Normalize to 0-255
        temp_min = np.min(self.thermal_image)
        temp_max = np.max(self.thermal_image)
        normalized = ((self.thermal_image - temp_min) / (temp_max - temp_min + 1e-10) * 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, colormap)
    
    def save(self, output_dir: str) -> str:
        """Save thermal data to disk"""
        import os
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_dir, f"thermal_{timestamp_str}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Save thermal data
        np.save(os.path.join(session_dir, "thermal_celsius.npy"), self.thermal_image)
        np.save(os.path.join(session_dir, "thermal_raw.npy"), self.thermal_raw)
        
        # Save colormap visualization
        colormap_img = self.get_colormap_image()
        cv2.imwrite(os.path.join(session_dir, "thermal_colormap.png"), colormap_img)
        
        # Save sequence if available
        if self.thermal_sequence:
            for i, frame in enumerate(self.thermal_sequence):
                np.save(os.path.join(session_dir, f"thermal_seq_{i:03d}.npy"), frame)
        
        # Save metadata
        import json
        with open(os.path.join(session_dir, "metadata.json"), 'w') as f:
            json.dump({
                'timestamp': self.timestamp.isoformat(),
                'ambient_temp': self.ambient_temp,
                'body_location': self.body_location,
                'patient_id': self.patient_id,
                'temp_range': {'min': float(np.min(self.thermal_image)), 
                              'max': float(np.max(self.thermal_image))},
                **self.metadata
            }, f, indent=2)
        
        return session_dir


@dataclass
class ThermalFeatures:
    """Extracted thermal features for a lesion"""
    # Temperature metrics
    mean_temp: float
    max_temp: float
    min_temp: float
    temp_range: float
    temp_std: float
    
    # Differential analysis
    delta_T: float  # Difference from surrounding tissue
    thermal_asymmetry: float
    hot_spot_ratio: float
    
    # Vascular patterns
    vascular_index: float
    perfusion_score: float
    
    # Dynamic recovery (if sequence available)
    recovery_rate: float
    recovery_time_constant: float
    thermal_inertia: float
    
    # Spatial patterns
    thermal_gradient: float
    edge_sharpness: float
    core_periphery_ratio: float


class ThermalImagingSystem:
    """Complete thermal imaging and analysis system"""
    
    def __init__(self, simulation_mode: bool = True):
        """
        Initialize thermal imaging system.
        
        Args:
            simulation_mode: If True, generate synthetic thermal data
        """
        self.simulation_mode = simulation_mode
        self.camera = None
        self.camera_available = False
        
        if not simulation_mode:
            self._initialize_camera()
        else:
            print("🌡️  Thermal imaging in simulation mode")
    
    def _initialize_camera(self):
        """Initialize FLIR camera"""
        try:
            from flirpy.camera.lepton import Lepton
            self.camera = Lepton()
            self.camera_available = True
            print("✓ FLIR Lepton camera initialized")
        except ImportError:
            print("⚠️  flirpy not installed - using simulation mode")
            self.simulation_mode = True
        except Exception as e:
            print(f"⚠️  FLIR camera not detected: {e}")
            self.simulation_mode = True
    
    def _generate_synthetic_thermal(self, 
                                    size: Tuple[int, int] = (120, 160),
                                    has_lesion: bool = True,
                                    lesion_type: str = "scc") -> np.ndarray:
        """
        Generate synthetic thermal image for testing.
        
        Args:
            size: Image dimensions (height, width)
            has_lesion: Whether to include a lesion
            lesion_type: Type of lesion to simulate
            
        Returns:
            Temperature image in Celsius
        """
        h, w = size
        
        # Base skin temperature with gradient
        base_temp = 34.0  # Normal skin temperature
        
        # Create smooth temperature gradient (body heat distribution)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        gradient = (y_coords / h - 0.5) * 1.0  # Vertical gradient
        
        # Add some random variation (blood flow patterns)
        np.random.seed(42)  # Reproducible
        noise = np.random.normal(0, 0.3, (h, w))
        smooth_noise = cv2.GaussianBlur(noise.astype(np.float32), (11, 11), 0)
        
        thermal = base_temp + gradient + smooth_noise
        
        if has_lesion:
            # Create lesion with elevated temperature
            center = (w // 2, h // 2)
            
            if lesion_type == "scc":
                # SCC: localized hot spot with irregular shape
                lesion_temp_increase = 1.5 + np.random.uniform(-0.3, 0.3)
                
                # Irregular lesion shape
                angles = np.linspace(0, 2 * np.pi, 50)
                radii = 15 + 5 * np.sin(4 * angles) + 3 * np.random.randn(50)
                radii = np.clip(radii, 8, 25)
                
                mask = np.zeros((h, w), dtype=np.float32)
                pts = np.array([
                    [int(center[0] + r * np.cos(a)), int(center[1] + r * np.sin(a))]
                    for a, r in zip(angles, radii)
                ], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1.0)
                
                # Smooth edges
                mask = cv2.GaussianBlur(mask, (9, 9), 0)
                
                # Apply temperature increase
                thermal += mask * lesion_temp_increase
                
                # Add hot core
                core_mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(core_mask, center, 8, 1.0, -1)
                core_mask = cv2.GaussianBlur(core_mask, (5, 5), 0)
                thermal += core_mask * 0.5  # Extra hot core
            
            elif lesion_type == "benign":
                # Benign: more uniform, less temperature difference
                lesion_temp_increase = 0.3
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(mask, center, 12, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
                thermal += mask * lesion_temp_increase
        
        return thermal
    
    def capture_thermal_snapshot(self) -> ThermalCapture:
        """Capture single thermal frame"""
        if self.simulation_mode:
            thermal = self._generate_synthetic_thermal(has_lesion=True)
            return ThermalCapture(
                thermal_image=thermal,
                thermal_raw=thermal,
                ambient_temp=25.0,
                metadata={'simulated': True}
            )
        
        # Real camera capture
        frame = self.camera.grab()
        
        # Convert to Celsius (camera-specific calibration)
        thermal_celsius = self._convert_to_celsius(frame)
        
        return ThermalCapture(
            thermal_image=thermal_celsius,
            thermal_raw=frame,
            ambient_temp=self._measure_ambient(),
            metadata={'simulated': False}
        )
    
    def capture_thermal_sequence(self, 
                                 duration: int = 30,
                                 fps: int = 2,
                                 cooling_stimulus: bool = True) -> ThermalCapture:
        """
        Capture thermal sequence with optional cooling stimulus.
        
        Args:
            duration: Recording duration in seconds
            fps: Frames per second
            cooling_stimulus: Apply cooling before recording
            
        Returns:
            ThermalCapture with sequence data
        """
        print("\n" + "=" * 50)
        print("    THERMAL IMAGING SEQUENCE")
        print("=" * 50)
        
        # Initial baseline
        print("\n[1/3] Capturing baseline thermal image...")
        baseline = self.capture_thermal_snapshot()
        
        if cooling_stimulus:
            print("\n[2/3] Cooling stimulus phase")
            print("    Instructions: Apply cool compress (15-20°C) for 10 seconds")
            print("    This reveals differential thermal recovery patterns")
            input("    Press Enter when ready to start recording...")
        else:
            print("\n[2/3] Skipping cooling stimulus")
        
        # Capture sequence
        print(f"\n[3/3] Recording thermal recovery ({duration}s at {fps} fps)...")
        sequence = []
        total_frames = duration * fps
        
        for i in range(total_frames):
            frame = self.capture_thermal_snapshot()
            sequence.append(frame.thermal_image)
            
            # Progress
            progress = (i + 1) / total_frames * 100
            print(f"    Recording: {progress:.0f}% ({i+1}/{total_frames} frames)", end='\r')
            
            time.sleep(1.0 / fps)
        
        print("\n✓ Thermal sequence capture complete")
        
        return ThermalCapture(
            thermal_image=baseline.thermal_image,
            thermal_raw=baseline.thermal_raw,
            thermal_sequence=sequence,
            ambient_temp=baseline.ambient_temp,
            timestamp=datetime.now(),
            metadata={
                'duration': duration,
                'fps': fps,
                'cooling_stimulus': cooling_stimulus,
                'total_frames': len(sequence)
            }
        )
    
    def _convert_to_celsius(self, raw_frame: np.ndarray) -> np.ndarray:
        """Convert raw sensor data to Celsius (camera-specific)"""
        # For FLIR Lepton: already in centikelvin
        # Convert to Celsius
        celsius = raw_frame / 100.0 - 273.15
        return celsius
    
    def _measure_ambient(self) -> float:
        """Measure ambient temperature"""
        # Use camera's ambient sensor if available
        # Otherwise return default
        return 25.0
    
    def extract_thermal_features(self, 
                                 thermal_capture: ThermalCapture,
                                 lesion_mask: Optional[np.ndarray] = None) -> ThermalFeatures:
        """
        Extract comprehensive thermal features.
        
        Args:
            thermal_capture: Captured thermal data
            lesion_mask: Optional mask of lesion region
            
        Returns:
            ThermalFeatures with all extracted metrics
        """
        thermal = thermal_capture.thermal_image
        
        # Auto-detect lesion if no mask provided
        if lesion_mask is None:
            lesion_mask = self._auto_detect_thermal_lesion(thermal)
        
        # Resize mask if needed
        if lesion_mask.shape != thermal.shape:
            lesion_mask = cv2.resize(lesion_mask, (thermal.shape[1], thermal.shape[0]))
        
        # Ensure binary mask
        lesion_mask = (lesion_mask > 127).astype(np.uint8) * 255
        
        # Temperature metrics within lesion
        lesion_temps = thermal[lesion_mask > 0]
        
        if len(lesion_temps) == 0:
            lesion_temps = thermal.flatten()
        
        mean_temp = float(np.mean(lesion_temps))
        max_temp = float(np.max(lesion_temps))
        min_temp = float(np.min(lesion_temps))
        temp_range = max_temp - min_temp
        temp_std = float(np.std(lesion_temps))
        
        # Surrounding tissue for comparison
        dilated = cv2.dilate(lesion_mask, np.ones((20, 20), np.uint8))
        surrounding_mask = dilated - lesion_mask
        surrounding_temps = thermal[surrounding_mask > 0]
        
        if len(surrounding_temps) > 0:
            delta_T = mean_temp - float(np.mean(surrounding_temps))
        else:
            delta_T = 0.0
        
        # Thermal asymmetry (compare left-right)
        h, w = thermal.shape
        left_temp = np.mean(thermal[:, :w//2])
        right_temp = np.mean(thermal[:, w//2:])
        thermal_asymmetry = abs(left_temp - right_temp)
        
        # Hot spot ratio
        threshold = np.percentile(thermal, 90)
        hot_spot_ratio = np.sum(lesion_mask[thermal > threshold] > 0) / (np.sum(lesion_mask > 0) + 1e-10)
        
        # Vascular patterns
        vascular_index = self._calculate_vascular_index(thermal, lesion_mask)
        perfusion_score = self._calculate_perfusion_score(thermal, lesion_mask)
        
        # Dynamic recovery analysis
        if thermal_capture.thermal_sequence:
            recovery_rate, time_constant, thermal_inertia = self._analyze_thermal_dynamics(
                thermal_capture.thermal_sequence, lesion_mask
            )
        else:
            recovery_rate = 0.0
            time_constant = 0.0
            thermal_inertia = 0.0
        
        # Spatial patterns
        thermal_gradient = self._calculate_thermal_gradient(thermal, lesion_mask)
        edge_sharpness = self._calculate_edge_sharpness(thermal, lesion_mask)
        core_periphery = self._calculate_core_periphery_ratio(thermal, lesion_mask)
        
        return ThermalFeatures(
            mean_temp=mean_temp,
            max_temp=max_temp,
            min_temp=min_temp,
            temp_range=temp_range,
            temp_std=temp_std,
            delta_T=delta_T,
            thermal_asymmetry=thermal_asymmetry,
            hot_spot_ratio=hot_spot_ratio,
            vascular_index=vascular_index,
            perfusion_score=perfusion_score,
            recovery_rate=recovery_rate,
            recovery_time_constant=time_constant,
            thermal_inertia=thermal_inertia,
            thermal_gradient=thermal_gradient,
            edge_sharpness=edge_sharpness,
            core_periphery_ratio=core_periphery
        )
    
    def _auto_detect_thermal_lesion(self, thermal: np.ndarray) -> np.ndarray:
        """Auto-detect lesion from thermal image based on temperature anomaly"""
        # Normalize for visualization
        thermal_norm = ((thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-10) * 255).astype(np.uint8)
        
        # Threshold high temperature regions
        threshold = np.percentile(thermal_norm, 85)
        _, mask = cv2.threshold(thermal_norm, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)
        
        return mask
    
    def _calculate_vascular_index(self, thermal: np.ndarray, mask: np.ndarray) -> float:
        """Calculate vascular index from thermal patterns"""
        # High-pass filter to detect vascular patterns
        blurred = cv2.GaussianBlur(thermal.astype(np.float32), (15, 15), 0)
        high_freq = thermal - blurred
        
        # Within lesion
        lesion_high_freq = high_freq[mask > 0]
        
        if len(lesion_high_freq) == 0:
            return 0.0
        
        # Vascular index = variation in high-frequency component
        return float(np.std(lesion_high_freq) / (np.mean(thermal) + 1e-10))
    
    def _calculate_perfusion_score(self, thermal: np.ndarray, mask: np.ndarray) -> float:
        """Calculate perfusion score based on temperature elevation"""
        lesion_temps = thermal[mask > 0]
        
        # Background temperature (25th percentile of image)
        background_temp = np.percentile(thermal, 25)
        
        if len(lesion_temps) == 0:
            return 0.0
        
        # Perfusion correlates with temperature elevation
        elevation = np.mean(lesion_temps) - background_temp
        
        return max(0.0, elevation / 2.0)  # Normalize to ~0-1 for typical elevations
    
    def _analyze_thermal_dynamics(self, 
                                  sequence: List[np.ndarray],
                                  mask: np.ndarray) -> Tuple[float, float, float]:
        """
        Analyze thermal recovery dynamics after cooling stimulus.
        
        Returns:
            (recovery_rate, time_constant, thermal_inertia)
        """
        if len(sequence) < 5:
            return 0.0, 0.0, 0.0
        
        # Extract mean temperature in lesion over time
        temps = []
        for frame in sequence:
            if frame.shape != mask.shape:
                frame_resized = cv2.resize(frame, (mask.shape[1], mask.shape[0]))
            else:
                frame_resized = frame
            
            lesion_temp = frame_resized[mask > 0]
            if len(lesion_temp) > 0:
                temps.append(np.mean(lesion_temp))
            else:
                temps.append(np.mean(frame_resized))
        
        temps = np.array(temps)
        
        if len(temps) < 3:
            return 0.0, 0.0, 0.0
        
        # Calculate recovery rate (slope of temperature recovery)
        t = np.arange(len(temps))
        
        # Fit exponential recovery: T(t) = T_final - (T_final - T_init) * exp(-t/tau)
        # Approximate with linear fit to log
        try:
            T_final = temps[-1]
            T_min = np.min(temps)
            
            if T_final > T_min:
                normalized = (T_final - temps) / (T_final - T_min + 1e-10)
                normalized = np.clip(normalized, 0.01, 0.99)
                log_norm = np.log(normalized)
                
                # Linear fit to get time constant
                coeffs = np.polyfit(t, log_norm, 1)
                time_constant = -1.0 / (coeffs[0] + 1e-10)
                
                # Recovery rate (initial slope in °C/s, assuming 2 fps)
                recovery_rate = (temps[2] - temps[0]) / (2 * 0.5)  # 2 frames at 0.5s each
                
                # Thermal inertia (higher = slower to change)
                thermal_inertia = time_constant / 10.0  # Normalize
            else:
                recovery_rate = 0.0
                time_constant = 0.0
                thermal_inertia = 0.0
        except:
            recovery_rate = 0.0
            time_constant = 0.0
            thermal_inertia = 0.0
        
        return float(recovery_rate), float(time_constant), float(thermal_inertia)
    
    def _calculate_thermal_gradient(self, thermal: np.ndarray, mask: np.ndarray) -> float:
        """Calculate thermal gradient magnitude within lesion"""
        # Sobel gradients
        grad_x = cv2.Sobel(thermal.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(thermal.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        lesion_gradient = gradient_mag[mask > 0]
        
        if len(lesion_gradient) == 0:
            return 0.0
        
        return float(np.mean(lesion_gradient))
    
    def _calculate_edge_sharpness(self, thermal: np.ndarray, mask: np.ndarray) -> float:
        """Calculate sharpness of thermal boundary"""
        # Get contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        contour = max(contours, key=cv2.contourArea)
        
        # Sample temperature gradient along contour
        gradients = []
        for point in contour[:, 0, :]:
            x, y = point
            
            # Get local gradient perpendicular to contour
            if 5 < x < thermal.shape[1] - 5 and 5 < y < thermal.shape[0] - 5:
                local_region = thermal[y-5:y+5, x-5:x+5]
                local_grad = np.max(local_region) - np.min(local_region)
                gradients.append(local_grad)
        
        if len(gradients) == 0:
            return 0.0
        
        return float(np.mean(gradients))
    
    def _calculate_core_periphery_ratio(self, thermal: np.ndarray, mask: np.ndarray) -> float:
        """Calculate ratio of core to periphery temperature"""
        # Erode mask to get core
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        core_mask = cv2.erode(mask, kernel)
        
        # Periphery = original - core
        periphery_mask = mask - core_mask
        
        core_temps = thermal[core_mask > 0]
        periphery_temps = thermal[periphery_mask > 0]
        
        if len(core_temps) == 0 or len(periphery_temps) == 0:
            return 1.0
        
        return float(np.mean(core_temps) / (np.mean(periphery_temps) + 1e-10))
    
    def generate_thermal_feature_vector(self, features: ThermalFeatures) -> np.ndarray:
        """Generate flat feature vector from thermal features"""
        return np.array([
            features.mean_temp,
            features.max_temp,
            features.min_temp,
            features.temp_range,
            features.temp_std,
            features.delta_T,
            features.thermal_asymmetry,
            features.hot_spot_ratio,
            features.vascular_index,
            features.perfusion_score,
            features.recovery_rate,
            features.recovery_time_constant,
            features.thermal_inertia,
            features.thermal_gradient,
            features.edge_sharpness,
            features.core_periphery_ratio
        ], dtype=np.float32)
    
    @staticmethod
    def get_thermal_feature_names() -> List[str]:
        """Get names for thermal feature vector"""
        return [
            'thermal_mean_temp', 'thermal_max_temp', 'thermal_min_temp',
            'thermal_temp_range', 'thermal_temp_std', 'thermal_delta_T',
            'thermal_asymmetry', 'thermal_hot_spot_ratio',
            'thermal_vascular_index', 'thermal_perfusion_score',
            'thermal_recovery_rate', 'thermal_time_constant', 'thermal_inertia',
            'thermal_gradient', 'thermal_edge_sharpness', 'thermal_core_periphery'
        ]

