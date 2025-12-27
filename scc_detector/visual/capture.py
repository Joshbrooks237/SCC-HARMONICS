"""
Multi-Spectrum Visual Capture System
=====================================

Comprehensive visual data acquisition across:
- Standard RGB (400-700nm)
- Polarized light imaging
- Dermoscopy (10x magnification)
- UV photography (320-400nm)
- Multispectral imaging

    "The naked eye sees color and shape.
     The dermoscope reveals structure.
     Polarization separates surface from depth.
     UV fluorescence exposes metabolic sin.
     
     We are no longer limited to mortal sight."

THE FIRST INTERROGATION BEGINS.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - The optical arsenal
# ═══════════════════════════════════════════════════════════════════════════════

import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
import os


@dataclass
class VisualCapture:
    """Complete visual spectrum data container"""
    rgb_standard: np.ndarray
    rgb_polarized_parallel: Optional[np.ndarray] = None
    rgb_polarized_cross: Optional[np.ndarray] = None
    dermoscopy: Optional[np.ndarray] = None
    uv_image: Optional[np.ndarray] = None
    multispectral: Dict[str, np.ndarray] = field(default_factory=dict)
    hdr_stack: List[np.ndarray] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    patient_id: str = ""
    lesion_id: str = ""
    body_location: str = ""
    
    def save(self, output_dir: str) -> str:
        """Save all captured data to disk"""
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(output_dir, f"{self.patient_id}_{timestamp_str}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Save main images
        cv2.imwrite(os.path.join(session_dir, "rgb_standard.png"), self.rgb_standard)
        
        if self.rgb_polarized_parallel is not None:
            cv2.imwrite(os.path.join(session_dir, "polarized_parallel.png"), 
                       self.rgb_polarized_parallel)
        
        if self.rgb_polarized_cross is not None:
            cv2.imwrite(os.path.join(session_dir, "polarized_cross.png"), 
                       self.rgb_polarized_cross)
        
        if self.dermoscopy is not None:
            cv2.imwrite(os.path.join(session_dir, "dermoscopy.png"), self.dermoscopy)
        
        if self.uv_image is not None:
            cv2.imwrite(os.path.join(session_dir, "uv_fluorescence.png"), self.uv_image)
        
        # Save multispectral channels
        for name, channel in self.multispectral.items():
            np.save(os.path.join(session_dir, f"multispectral_{name}.npy"), channel)
        
        # Save metadata
        import json
        metadata_path = os.path.join(session_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp.isoformat(),
                'patient_id': self.patient_id,
                'lesion_id': self.lesion_id,
                'body_location': self.body_location,
                'quality_metrics': self.metadata
            }, f, indent=2)
        
        return session_dir


class MultiSpectrumVisualCapture:
    """Comprehensive visual data acquisition system"""
    
    def __init__(self, camera_device: int = 0, simulation_mode: bool = False):
        """
        Initialize the visual capture system.
        
        Args:
            camera_device: Camera device index (default 0)
            simulation_mode: If True, generate synthetic data for testing
        """
        self.simulation_mode = simulation_mode
        self.camera_device = camera_device
        self.camera = None
        self.camera_available = False
        
        if not simulation_mode:
            self._initialize_camera()
        else:
            print("🔬 Running in simulation mode - generating synthetic lesion data")
    
    def _initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.camera = cv2.VideoCapture(self.camera_device)
            if self.camera.isOpened():
                self.camera_available = True
                self._setup_camera_optimal_settings()
                print("✓ Camera initialized successfully")
            else:
                print("⚠️  Camera not available - using simulation mode")
                self.simulation_mode = True
        except Exception as e:
            print(f"⚠️  Camera initialization failed: {e}")
            self.simulation_mode = True
    
    def _setup_camera_optimal_settings(self):
        """Configure camera for medical imaging"""
        if not self.camera_available:
            return
        
        # High resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 3072)
        
        # Manual controls for consistent imaging
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Manual exposure
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Manual focus
        self.camera.set(cv2.CAP_PROP_AUTO_WB, 0)  # Manual white balance
        
        # High quality
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    def _generate_synthetic_lesion(self, lesion_type: str = "scc") -> np.ndarray:
        """Generate synthetic lesion image for testing"""
        # Create base skin texture
        size = 512
        image = np.ones((size, size, 3), dtype=np.uint8) * 200  # Base skin tone
        
        # Add skin texture noise
        noise = np.random.normal(0, 10, (size, size, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Create lesion shape (irregular for SCC)
        center = (size // 2, size // 2)
        
        if lesion_type == "scc":
            # Irregular SCC-like lesion
            angles = np.linspace(0, 2 * np.pi, 100)
            radii = 80 + 30 * np.sin(5 * angles) + 20 * np.random.randn(100)
            radii = np.clip(radii, 30, 150)
            
            pts = np.array([
                [int(center[0] + r * np.cos(a)), int(center[1] + r * np.sin(a))]
                for a, r in zip(angles, radii)
            ], dtype=np.int32)
            
            # Create mask
            mask = np.zeros((size, size), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            
            # Lesion colors (reddish-brown with variation)
            lesion_color = np.array([60, 80, 140])  # BGR
            color_variation = np.random.normal(0, 15, (size, size, 3))
            
            # Apply lesion
            lesion_region = np.ones((size, size, 3)) * lesion_color + color_variation
            lesion_region = np.clip(lesion_region, 0, 255).astype(np.uint8)
            
            # Add scaling/crusting texture
            texture = np.random.randint(0, 40, (size, size, 3), dtype=np.uint8)
            lesion_region = cv2.add(lesion_region, texture)
            
            # Blend lesion with skin
            mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
            image = (image * (1 - mask_3ch) + lesion_region * mask_3ch).astype(np.uint8)
            
            # Add border irregularity
            kernel = np.ones((5, 5), np.uint8)
            edge_mask = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
            edge_color = np.array([40, 50, 100])  # Darker border
            edge_3ch = cv2.merge([edge_mask, edge_mask, edge_mask]) / 255.0
            edge_region = np.ones((size, size, 3)) * edge_color
            image = (image * (1 - edge_3ch * 0.7) + edge_region * edge_3ch * 0.7).astype(np.uint8)
        
        return image
    
    def guided_capture_sequence(self, 
                                patient_id: str = "TEST",
                                lesion_id: str = "L001",
                                body_location: str = "unknown") -> VisualCapture:
        """
        Interactive guided capture with real-time feedback.
        
        Returns complete VisualCapture with all modalities.
        """
        print("\n" + "=" * 60)
        print("    VISUAL SPECTRUM CAPTURE SEQUENCE")
        print("    Multi-Modal Skin Lesion Documentation")
        print("=" * 60)
        
        if self.simulation_mode:
            return self._simulated_capture_sequence(patient_id, lesion_id, body_location)
        
        # 1. Standard RGB with HDR
        print("\n[1/6] Capturing standard RGB (HDR stack)...")
        rgb_stack = self.capture_hdr_stack()
        rgb_standard = self.merge_hdr(rgb_stack)
        
        # 2. Cross-polarized imaging
        print("\n[2/6] Cross-polarized light imaging")
        print("    Instructions: Attach cross-polarizer to camera and light source")
        input("    Press Enter when ready...")
        rgb_polarized_cross = self.capture_with_quality_check("Cross-Polarized")
        
        # 3. Parallel-polarized imaging
        print("\n[3/6] Parallel-polarized light imaging")
        print("    Instructions: Rotate polarizer to parallel position")
        input("    Press Enter when ready...")
        rgb_polarized_parallel = self.capture_with_quality_check("Parallel-Polarized")
        
        # 4. Dermoscopic imaging
        print("\n[4/6] Dermoscopic capture")
        print("    Instructions: Attach dermoscopy lens, apply gel to lesion")
        input("    Press Enter when ready...")
        dermoscopy = self.capture_with_quality_check("Dermoscopy")
        
        # 5. UV fluorescence imaging
        print("\n[5/6] UV fluorescence imaging")
        print("    Instructions: Turn on UV light (365nm), turn off ambient light")
        input("    Press Enter when ready...")
        uv_image = self.capture_uv_fluorescence()
        
        # 6. Extract multispectral channels
        print("\n[6/6] Processing multispectral channels...")
        multispectral = self.extract_multispectral_channels(rgb_standard)
        
        # Quality validation
        print("\n📊 Validating capture quality...")
        all_images = {
            'rgb': rgb_standard,
            'polarized_cross': rgb_polarized_cross,
            'polarized_parallel': rgb_polarized_parallel,
            'dermoscopy': dermoscopy,
            'uv': uv_image
        }
        quality_metrics = self.validate_capture_quality(all_images)
        
        if quality_metrics['overall_quality'] < 0.7:
            print(f"⚠️  WARNING: Low capture quality ({quality_metrics['overall_quality']:.2f})")
            retry = input("    Retry capture? (y/n): ")
            if retry.lower() == 'y':
                return self.guided_capture_sequence(patient_id, lesion_id, body_location)
        else:
            print(f"✓ High quality capture (score: {quality_metrics['overall_quality']:.2f})")
        
        return VisualCapture(
            rgb_standard=rgb_standard,
            rgb_polarized_parallel=rgb_polarized_parallel,
            rgb_polarized_cross=rgb_polarized_cross,
            dermoscopy=dermoscopy,
            uv_image=uv_image,
            multispectral=multispectral,
            hdr_stack=rgb_stack,
            metadata=quality_metrics,
            timestamp=datetime.now(),
            patient_id=patient_id,
            lesion_id=lesion_id,
            body_location=body_location
        )
    
    def _simulated_capture_sequence(self, 
                                    patient_id: str,
                                    lesion_id: str,
                                    body_location: str) -> VisualCapture:
        """Generate simulated capture for testing"""
        print("\n🔬 Generating synthetic lesion data...")
        
        # Generate base synthetic lesion
        rgb_standard = self._generate_synthetic_lesion("scc")
        
        # Generate variations for different modalities
        rgb_polarized_cross = self._apply_cross_polarization_effect(rgb_standard)
        rgb_polarized_parallel = self._apply_parallel_polarization_effect(rgb_standard)
        dermoscopy = self._simulate_dermoscopy(rgb_standard)
        uv_image = self._simulate_uv_fluorescence(rgb_standard)
        
        # Extract multispectral
        multispectral = self.extract_multispectral_channels(rgb_standard)
        
        print("✓ Synthetic data generated successfully")
        
        return VisualCapture(
            rgb_standard=rgb_standard,
            rgb_polarized_parallel=rgb_polarized_parallel,
            rgb_polarized_cross=rgb_polarized_cross,
            dermoscopy=dermoscopy,
            uv_image=uv_image,
            multispectral=multispectral,
            hdr_stack=[rgb_standard],
            metadata={'overall_quality': 0.95, 'simulated': True},
            timestamp=datetime.now(),
            patient_id=patient_id,
            lesion_id=lesion_id,
            body_location=body_location
        )
    
    def _apply_cross_polarization_effect(self, image: np.ndarray) -> np.ndarray:
        """Simulate cross-polarization effect (reduces specular reflection)"""
        # Reduce brightness, enhance subsurface detail
        result = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
        # Enhance local contrast
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return result
    
    def _apply_parallel_polarization_effect(self, image: np.ndarray) -> np.ndarray:
        """Simulate parallel-polarization effect (enhances surface reflection)"""
        # Increase specular highlights
        result = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        # Add slight blur to simulate surface scatter
        result = cv2.GaussianBlur(result, (3, 3), 0)
        return result
    
    def _simulate_dermoscopy(self, image: np.ndarray) -> np.ndarray:
        """Simulate dermoscopic magnification"""
        h, w = image.shape[:2]
        # Zoom into center
        zoom_factor = 2.0
        center_crop = image[
            int(h/4):int(3*h/4),
            int(w/4):int(3*w/4)
        ]
        # Resize back to original size
        result = cv2.resize(center_crop, (w, h), interpolation=cv2.INTER_CUBIC)
        # Enhance contrast
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return result
    
    def _simulate_uv_fluorescence(self, image: np.ndarray) -> np.ndarray:
        """Simulate UV fluorescence imaging"""
        # Extract blue channel and enhance
        b, g, r = cv2.split(image)
        # Create fluorescence pattern
        fluorescence = cv2.addWeighted(b, 0.7, g, 0.3, 0)
        # Apply threshold for fluorescent areas
        _, bright_areas = cv2.threshold(fluorescence, 100, 255, cv2.THRESH_BINARY)
        # Create pseudo-color UV image
        uv = cv2.merge([
            fluorescence,  # Blue channel - base fluorescence
            np.clip(fluorescence * 0.5, 0, 255).astype(np.uint8),  # Green - reduced
            np.clip(fluorescence * 0.2, 0, 255).astype(np.uint8)   # Red - minimal
        ])
        return uv
    
    def capture_hdr_stack(self, exposures: List[int] = [-2, -1, 0, 1, 2]) -> List[np.ndarray]:
        """Capture HDR stack at different exposures"""
        if self.simulation_mode:
            base = self._generate_synthetic_lesion("scc")
            return [cv2.convertScaleAbs(base, alpha=1.0 + exp*0.2) for exp in exposures]
        
        stack = []
        for exp in exposures:
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exp)
            time.sleep(0.1)  # Allow camera to adjust
            ret, frame = self.camera.read()
            if ret:
                stack.append(frame)
        return stack
    
    def merge_hdr(self, stack: List[np.ndarray]) -> np.ndarray:
        """Merge HDR stack into single tone-mapped image"""
        if len(stack) == 0:
            return None
        if len(stack) == 1:
            return stack[0]
        
        # Exposure fusion (Mertens)
        merge_mertens = cv2.createMergeMertens()
        hdr = merge_mertens.process(stack)
        return np.clip(hdr * 255, 0, 255).astype(np.uint8)
    
    def capture_with_quality_check(self, mode_name: str = "Standard") -> Optional[np.ndarray]:
        """Capture with real-time quality feedback"""
        if self.simulation_mode:
            return self._generate_synthetic_lesion("scc")
        
        print(f"\n    Live preview - {mode_name}")
        print("    Press SPACE when image quality is satisfactory")
        print("    Press ESC to skip this modality")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Calculate quality metrics
            blur_score = self.calculate_blur_score(frame)
            exposure_ok = self.check_exposure(frame)
            
            # Create display with quality overlay
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Resize for display if too large
            if w > 1280:
                scale = 1280 / w
                display = cv2.resize(display, (1280, int(h * scale)))
            
            # Quality indicators
            blur_color = (0, 255, 0) if blur_score > 100 else (0, 0, 255)
            exp_color = (0, 255, 0) if exposure_ok else (0, 0, 255)
            
            cv2.putText(display, f"Focus: {'SHARP' if blur_score > 100 else 'BLURRY'} ({blur_score:.0f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blur_color, 2)
            cv2.putText(display, f"Exposure: {'OK' if exposure_ok else 'ADJUST'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, exp_color, 2)
            cv2.putText(display, f"Mode: {mode_name}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Quality Check - SPACE to capture, ESC to skip', display)
            
            key = cv2.waitKey(1)
            if key == 32:  # Space bar
                if blur_score > 100 and exposure_ok:
                    cv2.destroyAllWindows()
                    print(f"    ✓ {mode_name} captured")
                    return frame
                else:
                    print("    ⚠️  Quality too low - adjust camera")
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                print(f"    ⚠️  {mode_name} skipped")
                return None
    
    def capture_uv_fluorescence(self) -> Optional[np.ndarray]:
        """UV fluorescence imaging with longer exposure"""
        if self.simulation_mode:
            base = self._generate_synthetic_lesion("scc")
            return self._simulate_uv_fluorescence(base)
        
        # Increase exposure for UV fluorescence
        self.camera.set(cv2.CAP_PROP_EXPOSURE, 3)
        return self.capture_with_quality_check("UV Fluorescence")
    
    def extract_multispectral_channels(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract RGB channels and derived spectral indices"""
        if image is None:
            return {}
        
        b, g, r = cv2.split(image)
        
        # Prevent division by zero
        r_float = r.astype(float) + 1e-10
        g_float = g.astype(float) + 1e-10
        b_float = b.astype(float) + 1e-10
        
        # Derived indices relevant to skin lesion analysis
        melanin_index = np.log(1.0 / r_float) / np.log(1.0 / g_float + 1e-10)
        erythema_index = r_float / g_float
        
        # Normalize indices to 0-255 for visualization
        melanin_norm = self._normalize_to_uint8(melanin_index)
        erythema_norm = self._normalize_to_uint8(erythema_index)
        
        return {
            'red': r,
            'green': g,
            'blue': b,
            'melanin_index': melanin_index,
            'melanin_norm': melanin_norm,
            'erythema_index': erythema_index,
            'erythema_norm': erythema_norm,
            'luminance': cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        }
    
    def _normalize_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to 0-255 uint8"""
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        if arr_max - arr_min < 1e-10:
            return np.zeros(arr.shape, dtype=np.uint8)
        normalized = (arr - arr_min) / (arr_max - arr_min) * 255
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    def validate_capture_quality(self, images: Dict) -> Dict:
        """Comprehensive quality validation across all modalities"""
        scores = {}
        
        for name, img in images.items():
            if img is None:
                scores[name] = 0.0
                continue
            
            # Multiple quality metrics
            blur = self.calculate_blur_score(img)
            contrast = self.calculate_contrast(img)
            noise = self.estimate_noise(img)
            
            # Combined quality score
            blur_score = min(1.0, blur / 200)
            contrast_score = min(1.0, contrast)
            noise_score = max(0.0, 1.0 - noise)
            
            score = blur_score * 0.4 + contrast_score * 0.3 + noise_score * 0.3
            scores[name] = round(score, 3)
        
        # Overall quality
        valid_scores = [s for s in scores.values() if s > 0]
        scores['overall_quality'] = round(np.mean(valid_scores) if valid_scores else 0.0, 3)
        
        return scores
    
    def calculate_blur_score(self, image: np.ndarray) -> float:
        """Laplacian variance for focus/blur detection"""
        if image is None:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """RMS contrast normalized to [0, 1]"""
        if image is None:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return np.std(gray) / 128.0
    
    def estimate_noise(self, image: np.ndarray) -> float:
        """Estimate image noise level"""
        if image is None:
            return 1.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # High-pass filter to isolate noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(float) - blurred.astype(float)
        
        return min(1.0, np.std(noise) / 20.0)
    
    def __del__(self):
        """Cleanup camera resources"""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()

