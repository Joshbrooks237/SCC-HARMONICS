"""
Temporal Change Detection for SCC Monitoring
=============================================

Tracks lesion evolution over time to detect:
- Size progression (growth rate)
- Color changes (darkening, erythema)
- Textural evolution
- Shape changes (border irregularity progression)
- New symptom development

The "E" in ABCDE criteria - Evolution is one of the strongest 
indicators of malignancy.

    "Time is the cruelest witness.
     It does not lie. It does not forget.
     The lesion that was not there last month.
     The border that has grown irregular.
     The color that has deepened.
     
     Every day you wait is another day the cancer grows."

THE TEMPORAL INTERROGATION: WHAT HAS CHANGED?
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - The temporal analysis toolkit
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import cv2
import json
import os


@dataclass
class LesionSnapshot:
    """Single time-point capture of a lesion"""
    timestamp: datetime
    image: np.ndarray
    mask: np.ndarray
    
    # Measurements
    area_mm2: float
    perimeter_mm: float
    diameter_mm: float
    
    # Color metrics (LAB color space)
    mean_L: float  # Lightness
    mean_a: float  # Green-red
    mean_b: float  # Blue-yellow
    color_std: float  # Color heterogeneity
    
    # Texture metrics
    contrast: float
    homogeneity: float
    
    # Shape metrics
    asymmetry: float
    border_irregularity: float
    circularity: float
    
    # Optional thermal/acoustic data
    thermal_delta_T: Optional[float] = None
    acoustic_thd: Optional[float] = None
    
    patient_id: str = ""
    lesion_id: str = ""
    body_location: str = ""
    notes: str = ""


@dataclass
class TemporalFeatures:
    """Features derived from temporal analysis"""
    # Growth metrics
    area_change_rate: float  # mm²/month
    diameter_change_rate: float  # mm/month
    volume_doubling_time: float  # months (estimated from 2D)
    perimeter_growth_rate: float
    
    # Growth pattern
    growth_pattern: str  # "exponential", "linear", "stable", "regression"
    growth_acceleration: float  # Second derivative
    
    # Color evolution
    color_change_rate: float  # ΔE/month in LAB space
    darkening_rate: float  # ΔL/month
    erythema_change: float  # Δa/month (redness)
    
    # Shape evolution
    asymmetry_change: float
    border_irregularity_change: float
    shape_stability: float  # How stable is overall shape
    
    # Texture evolution
    contrast_change: float
    heterogeneity_change: float
    
    # Risk indicators
    rapid_change_flag: bool  # Any metric exceeding threshold
    evolution_risk_score: float  # 0-1 composite score
    
    # Time span
    observation_period_days: int
    num_observations: int


class TemporalChangeDetector:
    """
    Detect and quantify lesion changes over time.
    
    Critical for the "E" (Evolution) in ABCDE criteria.
    Rapid evolution strongly suggests malignancy.
    """
    
    def __init__(self, 
                 data_directory: str = "./lesion_data",
                 mm_per_pixel: float = 0.1):
        """
        Initialize temporal change detector.
        
        Args:
            data_directory: Directory to store/load lesion history
            mm_per_pixel: Calibration factor for measurements
        """
        self.data_directory = data_directory
        self.mm_per_pixel = mm_per_pixel
        
        # Create data directory if needed
        os.makedirs(data_directory, exist_ok=True)
        
        # Thresholds for rapid change detection
        self.thresholds = {
            'area_growth_mm2_per_month': 10.0,  # >10mm² growth/month is concerning
            'diameter_growth_mm_per_month': 1.0,  # >1mm/month diameter growth
            'color_change_per_month': 5.0,  # ΔE > 5 per month
            'asymmetry_increase_per_month': 0.1,
            'border_irregularity_increase': 0.1
        }
    
    def create_snapshot(self, 
                       image: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       patient_id: str = "",
                       lesion_id: str = "",
                       thermal_data: Optional[Dict] = None,
                       acoustic_data: Optional[Dict] = None) -> LesionSnapshot:
        """
        Create a snapshot of current lesion state.
        
        Args:
            image: RGB/BGR image of lesion
            mask: Binary mask (will auto-segment if None)
            patient_id: Patient identifier
            lesion_id: Lesion identifier
            thermal_data: Optional thermal measurements
            acoustic_data: Optional acoustic measurements
            
        Returns:
            LesionSnapshot with all measurements
        """
        if mask is None:
            mask = self._auto_segment(image)
        
        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Size measurements
        area_pixels = np.sum(mask > 0)
        area_mm2 = area_pixels * (self.mm_per_pixel ** 2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter_pixels = cv2.arcLength(contour, True)
            perimeter_mm = perimeter_pixels * self.mm_per_pixel
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            diameter_mm = 2 * radius * self.mm_per_pixel
            
            # Circularity
            if area_pixels > 0:
                circularity = 4 * np.pi * area_pixels / (perimeter_pixels ** 2 + 1e-10)
            else:
                circularity = 0.0
        else:
            perimeter_mm = 0.0
            diameter_mm = 0.0
            circularity = 0.0
        
        # Color measurements (in LAB space)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        masked_l = l[mask > 0]
        masked_a = a[mask > 0]
        masked_b = b[mask > 0]
        
        if len(masked_l) > 0:
            mean_L = float(np.mean(masked_l))
            mean_a = float(np.mean(masked_a))
            mean_b = float(np.mean(masked_b))
            
            # Color heterogeneity (std in LAB space)
            color_std = float(np.sqrt(
                np.var(masked_l) + np.var(masked_a) + np.var(masked_b)
            ))
        else:
            mean_L = mean_a = mean_b = color_std = 0.0
        
        # Texture measurements
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast, homogeneity = self._calculate_texture_metrics(gray, mask)
        
        # Shape measurements
        asymmetry = self._calculate_asymmetry(mask)
        border_irreg = self._calculate_border_irregularity(mask)
        
        # Optional measurements
        thermal_delta_T = None
        acoustic_thd = None
        
        if thermal_data:
            thermal_delta_T = thermal_data.get('delta_T')
        
        if acoustic_data:
            acoustic_thd = acoustic_data.get('thd')
        
        return LesionSnapshot(
            timestamp=datetime.now(),
            image=image,
            mask=mask,
            area_mm2=float(area_mm2),
            perimeter_mm=float(perimeter_mm),
            diameter_mm=float(diameter_mm),
            mean_L=mean_L,
            mean_a=mean_a,
            mean_b=mean_b,
            color_std=color_std,
            contrast=float(contrast),
            homogeneity=float(homogeneity),
            asymmetry=float(asymmetry),
            border_irregularity=float(border_irreg),
            circularity=float(circularity),
            thermal_delta_T=thermal_delta_T,
            acoustic_thd=acoustic_thd,
            patient_id=patient_id,
            lesion_id=lesion_id
        )
    
    def _auto_segment(self, image: np.ndarray) -> np.ndarray:
        """Auto-segment lesion from image"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Otsu's thresholding on L channel
        _, mask = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up
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
    
    def _calculate_texture_metrics(self, 
                                   gray: np.ndarray, 
                                   mask: np.ndarray) -> Tuple[float, float]:
        """Calculate texture contrast and homogeneity"""
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            gray_masked = gray.copy()
            gray_masked[mask == 0] = 0
            gray_quantized = (gray_masked // 4).astype(np.uint8)
            
            glcm = graycomatrix(gray_quantized, [1], [0], 64, symmetric=True, normed=True)
            contrast = float(graycoprops(glcm, 'contrast')[0, 0])
            homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
            
            return contrast, homogeneity
        except ImportError:
            return 0.0, 0.0
    
    def _calculate_asymmetry(self, mask: np.ndarray) -> float:
        """Calculate shape asymmetry"""
        h, w = mask.shape
        
        # Horizontal asymmetry
        left = mask[:, :w//2]
        right = cv2.flip(mask[:, w//2:], 1)
        
        min_w = min(left.shape[1], right.shape[1])
        h_diff = np.sum(np.abs(left[:, -min_w:].astype(float) - 
                              right[:, :min_w].astype(float)))
        
        # Vertical asymmetry
        top = mask[:h//2, :]
        bottom = cv2.flip(mask[h//2:, :], 0)
        
        min_h = min(top.shape[0], bottom.shape[0])
        v_diff = np.sum(np.abs(top[-min_h:, :].astype(float) - 
                              bottom[:min_h, :].astype(float)))
        
        total_area = np.sum(mask) / 255.0 + 1e-10
        asymmetry = (h_diff + v_diff) / (2 * total_area * 255)
        
        return min(1.0, asymmetry)
    
    def _calculate_border_irregularity(self, mask: np.ndarray) -> float:
        """Calculate border irregularity"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if area == 0:
            return 0.0
        
        compactness = (perimeter ** 2) / area
        irregularity = (compactness - 4 * np.pi) / (4 * np.pi)
        
        return min(1.0, max(0.0, irregularity))
    
    def save_snapshot(self, snapshot: LesionSnapshot) -> str:
        """Save snapshot to disk"""
        # Create patient/lesion directory
        lesion_dir = os.path.join(
            self.data_directory,
            snapshot.patient_id,
            snapshot.lesion_id
        )
        os.makedirs(lesion_dir, exist_ok=True)
        
        # Filename with timestamp
        timestamp_str = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(lesion_dir, timestamp_str)
        
        # Save image and mask
        cv2.imwrite(f"{base_path}_image.png", snapshot.image)
        cv2.imwrite(f"{base_path}_mask.png", snapshot.mask)
        
        # Save measurements as JSON
        measurements = {
            'timestamp': snapshot.timestamp.isoformat(),
            'area_mm2': snapshot.area_mm2,
            'perimeter_mm': snapshot.perimeter_mm,
            'diameter_mm': snapshot.diameter_mm,
            'mean_L': snapshot.mean_L,
            'mean_a': snapshot.mean_a,
            'mean_b': snapshot.mean_b,
            'color_std': snapshot.color_std,
            'contrast': snapshot.contrast,
            'homogeneity': snapshot.homogeneity,
            'asymmetry': snapshot.asymmetry,
            'border_irregularity': snapshot.border_irregularity,
            'circularity': snapshot.circularity,
            'thermal_delta_T': snapshot.thermal_delta_T,
            'acoustic_thd': snapshot.acoustic_thd,
            'body_location': snapshot.body_location,
            'notes': snapshot.notes
        }
        
        with open(f"{base_path}_measurements.json", 'w') as f:
            json.dump(measurements, f, indent=2)
        
        return base_path
    
    def load_history(self, patient_id: str, lesion_id: str) -> List[LesionSnapshot]:
        """Load all snapshots for a lesion"""
        lesion_dir = os.path.join(self.data_directory, patient_id, lesion_id)
        
        if not os.path.exists(lesion_dir):
            return []
        
        snapshots = []
        
        # Find all measurement files
        for filename in sorted(os.listdir(lesion_dir)):
            if filename.endswith('_measurements.json'):
                base = filename.replace('_measurements.json', '')
                
                # Load data
                with open(os.path.join(lesion_dir, f"{base}_measurements.json")) as f:
                    data = json.load(f)
                
                image_path = os.path.join(lesion_dir, f"{base}_image.png")
                mask_path = os.path.join(lesion_dir, f"{base}_mask.png")
                
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    image = cv2.imread(image_path)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    snapshot = LesionSnapshot(
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        image=image,
                        mask=mask,
                        area_mm2=data['area_mm2'],
                        perimeter_mm=data['perimeter_mm'],
                        diameter_mm=data['diameter_mm'],
                        mean_L=data['mean_L'],
                        mean_a=data['mean_a'],
                        mean_b=data['mean_b'],
                        color_std=data['color_std'],
                        contrast=data['contrast'],
                        homogeneity=data['homogeneity'],
                        asymmetry=data['asymmetry'],
                        border_irregularity=data['border_irregularity'],
                        circularity=data['circularity'],
                        thermal_delta_T=data.get('thermal_delta_T'),
                        acoustic_thd=data.get('acoustic_thd'),
                        patient_id=patient_id,
                        lesion_id=lesion_id,
                        body_location=data.get('body_location', ''),
                        notes=data.get('notes', '')
                    )
                    snapshots.append(snapshot)
        
        return sorted(snapshots, key=lambda s: s.timestamp)
    
    def analyze_temporal_changes(self, 
                                snapshots: List[LesionSnapshot]) -> TemporalFeatures:
        """
        Analyze changes across multiple time points.
        
        Args:
            snapshots: List of snapshots sorted by time
            
        Returns:
            TemporalFeatures with change metrics
        """
        if len(snapshots) < 2:
            return self._empty_temporal_features(snapshots)
        
        # Calculate observation period
        first = snapshots[0]
        last = snapshots[-1]
        period_days = (last.timestamp - first.timestamp).days
        period_months = period_days / 30.0 + 1e-10
        
        # Extract time series
        timestamps = np.array([(s.timestamp - first.timestamp).days for s in snapshots])
        areas = np.array([s.area_mm2 for s in snapshots])
        diameters = np.array([s.diameter_mm for s in snapshots])
        perimeters = np.array([s.perimeter_mm for s in snapshots])
        
        # Color time series
        L_values = np.array([s.mean_L for s in snapshots])
        a_values = np.array([s.mean_a for s in snapshots])
        b_values = np.array([s.mean_b for s in snapshots])
        
        # Shape time series
        asymmetries = np.array([s.asymmetry for s in snapshots])
        border_irregs = np.array([s.border_irregularity for s in snapshots])
        
        # Texture time series
        contrasts = np.array([s.contrast for s in snapshots])
        homogeneities = np.array([s.homogeneity for s in snapshots])
        
        # Growth rates (linear regression)
        area_slope = self._linear_rate(timestamps, areas, period_months)
        diameter_slope = self._linear_rate(timestamps, diameters, period_months)
        perimeter_slope = self._linear_rate(timestamps, perimeters, period_months)
        
        # Volume doubling time (assuming hemisphere)
        volume_doubling = self._estimate_doubling_time(timestamps, areas)
        
        # Growth pattern classification
        growth_pattern, acceleration = self._classify_growth_pattern(timestamps, areas)
        
        # Color change rate (ΔE in LAB)
        delta_e = np.sqrt(
            (L_values[-1] - L_values[0])**2 +
            (a_values[-1] - a_values[0])**2 +
            (b_values[-1] - b_values[0])**2
        )
        color_change_rate = delta_e / period_months
        
        # Darkening rate
        darkening_rate = (L_values[0] - L_values[-1]) / period_months  # Positive = darkening
        
        # Erythema change
        erythema_change = (a_values[-1] - a_values[0]) / period_months  # Positive = more red
        
        # Shape evolution
        asymmetry_change = (asymmetries[-1] - asymmetries[0]) / period_months
        border_change = (border_irregs[-1] - border_irregs[0]) / period_months
        shape_stability = 1.0 - np.std(asymmetries) / (np.mean(asymmetries) + 1e-10)
        
        # Texture evolution
        contrast_change = (contrasts[-1] - contrasts[0]) / period_months
        heterogeneity_change = (1/homogeneities[-1] - 1/homogeneities[0]) / period_months
        
        # Rapid change detection
        rapid_change = (
            abs(area_slope) > self.thresholds['area_growth_mm2_per_month'] or
            abs(diameter_slope) > self.thresholds['diameter_growth_mm_per_month'] or
            color_change_rate > self.thresholds['color_change_per_month'] or
            asymmetry_change > self.thresholds['asymmetry_increase_per_month'] or
            border_change > self.thresholds['border_irregularity_increase']
        )
        
        # Composite evolution risk score
        risk_score = self._calculate_evolution_risk(
            area_slope, diameter_slope, color_change_rate,
            asymmetry_change, border_change, darkening_rate
        )
        
        return TemporalFeatures(
            area_change_rate=float(area_slope),
            diameter_change_rate=float(diameter_slope),
            volume_doubling_time=float(volume_doubling),
            perimeter_growth_rate=float(perimeter_slope),
            growth_pattern=growth_pattern,
            growth_acceleration=float(acceleration),
            color_change_rate=float(color_change_rate),
            darkening_rate=float(darkening_rate),
            erythema_change=float(erythema_change),
            asymmetry_change=float(asymmetry_change),
            border_irregularity_change=float(border_change),
            shape_stability=float(max(0, min(1, shape_stability))),
            contrast_change=float(contrast_change),
            heterogeneity_change=float(heterogeneity_change),
            rapid_change_flag=rapid_change,
            evolution_risk_score=float(risk_score),
            observation_period_days=period_days,
            num_observations=len(snapshots)
        )
    
    def _linear_rate(self, 
                     timestamps: np.ndarray, 
                     values: np.ndarray,
                     period_months: float) -> float:
        """Calculate linear rate of change (per month)"""
        if len(timestamps) < 2:
            return 0.0
        
        change = values[-1] - values[0]
        return change / period_months
    
    def _estimate_doubling_time(self, 
                               timestamps: np.ndarray, 
                               areas: np.ndarray) -> float:
        """Estimate volume doubling time from area growth"""
        if len(timestamps) < 2 or areas[0] <= 0:
            return float('inf')
        
        # Fit exponential growth
        try:
            log_areas = np.log(areas + 1e-10)
            slope, _ = np.polyfit(timestamps, log_areas, 1)
            
            if slope <= 0:
                return float('inf')
            
            # Time to double (in days) - for volume, multiply by 1.5 (sphere vs circle)
            doubling_days = np.log(2) / slope * 1.5
            return doubling_days / 30  # Convert to months
        except:
            return float('inf')
    
    def _classify_growth_pattern(self, 
                                timestamps: np.ndarray, 
                                areas: np.ndarray) -> Tuple[str, float]:
        """Classify growth pattern and calculate acceleration"""
        if len(timestamps) < 3:
            return "unknown", 0.0
        
        # Calculate first and second derivatives
        dt = np.diff(timestamps).astype(float)
        dt[dt == 0] = 1e-10
        
        da = np.diff(areas)
        first_deriv = da / dt
        
        if len(first_deriv) > 1:
            d2a = np.diff(first_deriv)
            dt2 = dt[1:]
            second_deriv = d2a / dt2
            acceleration = float(np.mean(second_deriv))
        else:
            acceleration = 0.0
        
        # Classify pattern
        mean_growth = np.mean(first_deriv)
        
        if abs(mean_growth) < 0.5:  # < 0.5 mm² per day
            pattern = "stable"
        elif mean_growth < 0:
            pattern = "regression"
        elif acceleration > 0.01:
            pattern = "exponential"
        else:
            pattern = "linear"
        
        return pattern, acceleration
    
    def _calculate_evolution_risk(self,
                                 area_rate: float,
                                 diameter_rate: float,
                                 color_rate: float,
                                 asymmetry_rate: float,
                                 border_rate: float,
                                 darkening_rate: float) -> float:
        """Calculate composite evolution risk score"""
        # Normalize each metric against thresholds
        area_risk = abs(area_rate) / self.thresholds['area_growth_mm2_per_month']
        diameter_risk = abs(diameter_rate) / self.thresholds['diameter_growth_mm_per_month']
        color_risk = color_rate / self.thresholds['color_change_per_month']
        asymmetry_risk = asymmetry_rate / self.thresholds['asymmetry_increase_per_month']
        border_risk = border_rate / self.thresholds['border_irregularity_increase']
        darkening_risk = max(0, darkening_rate) / 5.0  # Darkening is concerning
        
        # Weighted combination
        risk = (
            area_risk * 0.2 +
            diameter_risk * 0.2 +
            color_risk * 0.15 +
            asymmetry_risk * 0.15 +
            border_risk * 0.15 +
            darkening_risk * 0.15
        )
        
        return min(1.0, max(0.0, risk))
    
    def _empty_temporal_features(self, snapshots: List[LesionSnapshot]) -> TemporalFeatures:
        """Return empty features for insufficient data"""
        return TemporalFeatures(
            area_change_rate=0.0,
            diameter_change_rate=0.0,
            volume_doubling_time=float('inf'),
            perimeter_growth_rate=0.0,
            growth_pattern="unknown",
            growth_acceleration=0.0,
            color_change_rate=0.0,
            darkening_rate=0.0,
            erythema_change=0.0,
            asymmetry_change=0.0,
            border_irregularity_change=0.0,
            shape_stability=1.0,
            contrast_change=0.0,
            heterogeneity_change=0.0,
            rapid_change_flag=False,
            evolution_risk_score=0.0,
            observation_period_days=0,
            num_observations=len(snapshots)
        )
    
    def generate_temporal_feature_vector(self, features: TemporalFeatures) -> np.ndarray:
        """Generate flat feature vector for ML"""
        return np.array([
            features.area_change_rate,
            features.diameter_change_rate,
            min(features.volume_doubling_time, 120),  # Cap at 10 years
            features.perimeter_growth_rate,
            features.growth_acceleration,
            features.color_change_rate,
            features.darkening_rate,
            features.erythema_change,
            features.asymmetry_change,
            features.border_irregularity_change,
            features.shape_stability,
            features.contrast_change,
            features.heterogeneity_change,
            float(features.rapid_change_flag),
            features.evolution_risk_score,
            features.observation_period_days / 365.0,  # Years
            features.num_observations
        ], dtype=np.float32)
    
    @staticmethod
    def get_temporal_feature_names() -> List[str]:
        """Get names for temporal feature vector"""
        return [
            'temporal_area_change_rate',
            'temporal_diameter_change_rate',
            'temporal_volume_doubling_time',
            'temporal_perimeter_growth_rate',
            'temporal_growth_acceleration',
            'temporal_color_change_rate',
            'temporal_darkening_rate',
            'temporal_erythema_change',
            'temporal_asymmetry_change',
            'temporal_border_change',
            'temporal_shape_stability',
            'temporal_contrast_change',
            'temporal_heterogeneity_change',
            'temporal_rapid_change_flag',
            'temporal_evolution_risk_score',
            'temporal_observation_years',
            'temporal_num_observations'
        ]
    
    def generate_evolution_report(self, 
                                 features: TemporalFeatures,
                                 snapshots: List[LesionSnapshot]) -> str:
        """Generate human-readable evolution report"""
        report = []
        report.append("=" * 60)
        report.append("LESION EVOLUTION REPORT")
        report.append("=" * 60)
        
        # Observation period
        if snapshots:
            report.append(f"\n📅 Observation Period:")
            report.append(f"   First: {snapshots[0].timestamp.strftime('%Y-%m-%d')}")
            report.append(f"   Last: {snapshots[-1].timestamp.strftime('%Y-%m-%d')}")
            report.append(f"   Duration: {features.observation_period_days} days")
            report.append(f"   Observations: {features.num_observations}")
        
        # Risk assessment
        report.append(f"\n⚠️  EVOLUTION RISK SCORE: {features.evolution_risk_score:.2f}/1.00")
        
        if features.rapid_change_flag:
            report.append("   🚨 RAPID CHANGE DETECTED - RECOMMEND CLINICAL REVIEW")
        
        # Growth analysis
        report.append(f"\n📏 Size Changes:")
        report.append(f"   Area: {features.area_change_rate:+.2f} mm²/month")
        report.append(f"   Diameter: {features.diameter_change_rate:+.2f} mm/month")
        report.append(f"   Pattern: {features.growth_pattern}")
        
        if features.volume_doubling_time < 12:
            report.append(f"   ⚠️  Volume doubling time: {features.volume_doubling_time:.1f} months")
        
        # Color changes
        report.append(f"\n🎨 Color Changes:")
        report.append(f"   Overall change: {features.color_change_rate:.2f} ΔE/month")
        
        if features.darkening_rate > 0:
            report.append(f"   Darkening: {features.darkening_rate:.2f}/month ⚠️")
        
        if features.erythema_change > 0:
            report.append(f"   Increased redness: {features.erythema_change:.2f}/month")
        
        # Shape changes
        report.append(f"\n🔷 Shape Changes:")
        report.append(f"   Asymmetry: {features.asymmetry_change:+.3f}/month")
        report.append(f"   Border irregularity: {features.border_irregularity_change:+.3f}/month")
        report.append(f"   Shape stability: {features.shape_stability:.2f}")
        
        # Texture changes
        report.append(f"\n🔍 Texture Changes:")
        report.append(f"   Contrast: {features.contrast_change:+.2f}/month")
        report.append(f"   Heterogeneity: {features.heterogeneity_change:+.2f}/month")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

