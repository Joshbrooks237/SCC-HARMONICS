"""
Visual Feature Extraction for SCC Detection

Implements comprehensive feature extraction including:
- ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution)
- Texture features (GLCM, LBP)
- Color histogram analysis
- Polarization ratios
- UV fluorescence patterns
- Multispectral indices
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .capture import VisualCapture


@dataclass
class LesionFeatures:
    """Complete feature set for a lesion"""
    # ABCDE criteria
    asymmetry_score: float
    border_irregularity: float
    color_variation: float
    diameter_mm: float
    
    # Texture features
    glcm_features: Dict[str, float]
    lbp_histogram: np.ndarray
    
    # Color features
    color_histograms: Dict[str, np.ndarray]
    polarization_ratio: float
    
    # UV/Multispectral
    uv_features: Dict[str, float]
    melanin_features: Dict[str, float]
    vascular_features: Dict[str, float]
    
    # Derived risk indicators
    seven_point_score: float
    three_point_score: float
    
    # Raw mask for visualization
    lesion_mask: np.ndarray


class VisualFeatureExtractor:
    """Extract comprehensive medical features from visual data"""
    
    def __init__(self, mm_per_pixel: float = 0.1):
        """
        Initialize the feature extractor.
        
        Args:
            mm_per_pixel: Calibration factor for size measurements
        """
        self.mm_per_pixel = mm_per_pixel
        self.color_spaces = ['RGB', 'HSV', 'LAB', 'YCrCb']
    
    def extract_all_features(self, visual_capture: VisualCapture) -> LesionFeatures:
        """
        Comprehensive feature extraction from all visual modalities.
        
        Args:
            visual_capture: Complete visual capture data
            
        Returns:
            LesionFeatures with all extracted features
        """
        # Determine primary image for analysis
        primary_image = visual_capture.dermoscopy if visual_capture.dermoscopy is not None \
                       else visual_capture.rgb_standard
        
        # Segment lesion first
        lesion_mask = self.segment_lesion(primary_image)
        
        # ABCDE criteria
        asymmetry = self.calculate_asymmetry(primary_image, lesion_mask)
        border_irreg = self.calculate_border_irregularity(lesion_mask)
        color_var = self.calculate_color_variation(primary_image, lesion_mask)
        diameter = self.estimate_diameter(lesion_mask)
        
        # Texture features
        glcm = self.extract_glcm_features(primary_image, lesion_mask)
        lbp = self.extract_lbp_features(primary_image, lesion_mask)
        
        # Color features
        histograms = self.extract_color_histograms(primary_image, lesion_mask)
        polarization = self.calculate_polarization_ratio(
            visual_capture.rgb_polarized_parallel,
            visual_capture.rgb_polarized_cross,
            lesion_mask
        )
        
        # UV features
        uv_features = self.analyze_uv_fluorescence(visual_capture.uv_image, lesion_mask)
        
        # Multispectral indices
        melanin = self.analyze_melanin_distribution(visual_capture.multispectral, lesion_mask)
        vascular = self.analyze_vascular_patterns(visual_capture.multispectral, lesion_mask)
        
        # Clinical scoring systems
        seven_point = self.calculate_seven_point_checklist(
            asymmetry, border_irreg, color_var, glcm, vascular
        )
        three_point = self.calculate_three_point_checklist(
            asymmetry, color_var, vascular
        )
        
        return LesionFeatures(
            asymmetry_score=asymmetry,
            border_irregularity=border_irreg,
            color_variation=color_var,
            diameter_mm=diameter,
            glcm_features=glcm,
            lbp_histogram=lbp,
            color_histograms=histograms,
            polarization_ratio=polarization,
            uv_features=uv_features,
            melanin_features=melanin,
            vascular_features=vascular,
            seven_point_score=seven_point,
            three_point_score=three_point,
            lesion_mask=lesion_mask
        )
    
    def segment_lesion(self, image: np.ndarray) -> np.ndarray:
        """
        Automatic lesion segmentation using adaptive thresholding.
        
        Args:
            image: BGR image
            
        Returns:
            Binary mask of lesion (255 = lesion, 0 = background)
        """
        if image is None:
            return np.zeros((512, 512), dtype=np.uint8)
        
        # Convert to LAB color space (better for skin lesions)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive thresholding on L channel
        thresh = cv2.adaptiveThreshold(
            l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )
        
        # Alternative: Otsu's method on b channel (blue-yellow)
        _, otsu_thresh = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine both methods
        combined = cv2.bitwise_or(thresh, otsu_thresh)
        
        # Morphological operations to clean up
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # Keep only largest connected component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        return mask
    
    def calculate_asymmetry(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate asymmetry score (0 = symmetric, 1 = highly asymmetric).
        
        Measures asymmetry along both major axes.
        """
        if mask is None or np.sum(mask) == 0:
            return 0.0
        
        # Find centroid and principal axes
        moments = cv2.moments(mask)
        if moments['m00'] == 0:
            return 0.0
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        h, w = mask.shape
        
        # Horizontal asymmetry
        left_half = mask[:, :cx]
        right_half = mask[:, cx:]
        
        # Make same size by flipping and cropping
        right_flipped = cv2.flip(right_half, 1)
        min_w = min(left_half.shape[1], right_flipped.shape[1])
        
        if min_w > 0:
            left_crop = left_half[:, -min_w:]
            right_crop = right_flipped[:, :min_w]
            h_asymmetry = np.sum(np.abs(left_crop.astype(float) - right_crop.astype(float)))
        else:
            h_asymmetry = 0
        
        # Vertical asymmetry
        top_half = mask[:cy, :]
        bottom_half = mask[cy:, :]
        
        bottom_flipped = cv2.flip(bottom_half, 0)
        min_h = min(top_half.shape[0], bottom_flipped.shape[0])
        
        if min_h > 0:
            top_crop = top_half[-min_h:, :]
            bottom_crop = bottom_flipped[:min_h, :]
            v_asymmetry = np.sum(np.abs(top_crop.astype(float) - bottom_crop.astype(float)))
        else:
            v_asymmetry = 0
        
        # Normalize by mask area
        total_pixels = np.sum(mask) / 255.0
        if total_pixels == 0:
            return 0.0
        
        asymmetry = (h_asymmetry + v_asymmetry) / (2 * total_pixels * 255)
        
        return min(1.0, asymmetry)
    
    def calculate_border_irregularity(self, mask: np.ndarray) -> float:
        """
        Calculate border irregularity score (0 = smooth, 1 = highly irregular).
        
        Based on compactness ratio and fractal dimension approximation.
        """
        if mask is None or np.sum(mask) == 0:
            return 0.0
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        contour = max(contours, key=cv2.contourArea)
        
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if area == 0:
            return 0.0
        
        # Compactness: circle has value of 4π ≈ 12.57
        compactness = (perimeter ** 2) / area
        min_compactness = 4 * np.pi  # Perfect circle
        
        # Normalized irregularity
        irregularity = (compactness - min_compactness) / min_compactness
        
        # Additional: convexity defect analysis
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3 and len(contour) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                # Count significant defects
                significant_defects = np.sum(defects[:, 0, 3] > 1000)
                irregularity += significant_defects * 0.05
        
        return min(1.0, irregularity)
    
    def calculate_color_variation(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate color variation within the lesion (0 = uniform, 1 = highly varied).
        
        Analyzes variation in HSV and LAB color spaces.
        """
        if image is None or mask is None or np.sum(mask) == 0:
            return 0.0
        
        # Extract pixels within mask
        masked_pixels = image[mask > 0]
        
        if len(masked_pixels) < 10:
            return 0.0
        
        # Convert to HSV
        hsv_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        
        # Hue variation (circular statistic)
        hue = hsv_pixels[:, 0, 0].astype(float)
        hue_rad = hue * 2 * np.pi / 180
        mean_sin = np.mean(np.sin(hue_rad))
        mean_cos = np.mean(np.cos(hue_rad))
        hue_variance = 1 - np.sqrt(mean_sin**2 + mean_cos**2)
        
        # Saturation and Value variation
        sat_std = np.std(hsv_pixels[:, 0, 1]) / 128.0
        val_std = np.std(hsv_pixels[:, 0, 2]) / 128.0
        
        # Convert to LAB for perceptual color difference
        lab_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
        
        # Color diversity using histogram entropy
        hist_entropy = 0
        for i in range(3):
            hist, _ = np.histogram(lab_pixels[:, 0, i], bins=32, range=(0, 256))
            hist = hist / hist.sum() + 1e-10
            hist_entropy -= np.sum(hist * np.log2(hist))
        hist_entropy /= (3 * 5)  # Normalize by max entropy (log2(32) ≈ 5)
        
        # Combined score
        color_variation = (hue_variance * 0.4 + sat_std * 0.2 + val_std * 0.2 + hist_entropy * 0.2)
        
        return min(1.0, color_variation)
    
    def estimate_diameter(self, mask: np.ndarray) -> float:
        """
        Estimate lesion diameter in millimeters.
        
        Uses minimum enclosing circle and calibration factor.
        """
        if mask is None or np.sum(mask) == 0:
            return 0.0
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        contour = max(contours, key=cv2.contourArea)
        
        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter_pixels = radius * 2
        diameter_mm = diameter_pixels * self.mm_per_pixel
        
        return diameter_mm
    
    def extract_glcm_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract Gray Level Co-occurrence Matrix texture features.
        
        Features: contrast, dissimilarity, homogeneity, energy, correlation, ASM
        """
        if image is None or mask is None:
            return self._empty_glcm_features()
        
        try:
            from skimage.feature import graycomatrix, graycoprops
        except ImportError:
            print("⚠️  skimage not available - returning empty GLCM features")
            return self._empty_glcm_features()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply mask - set background to 0
        gray_masked = gray.copy()
        gray_masked[mask == 0] = 0
        
        # Reduce levels for computational efficiency
        gray_quantized = (gray_masked // 4).astype(np.uint8)  # 64 levels
        
        # Calculate GLCM at multiple distances and angles
        distances = [1, 2, 4]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(
            gray_quantized, 
            distances=distances, 
            angles=angles,
            levels=64, 
            symmetric=True, 
            normed=True
        )
        
        # Extract properties
        features = {
            'contrast': float(graycoprops(glcm, 'contrast').mean()),
            'dissimilarity': float(graycoprops(glcm, 'dissimilarity').mean()),
            'homogeneity': float(graycoprops(glcm, 'homogeneity').mean()),
            'energy': float(graycoprops(glcm, 'energy').mean()),
            'correlation': float(graycoprops(glcm, 'correlation').mean()),
            'ASM': float(graycoprops(glcm, 'ASM').mean())
        }
        
        return features
    
    def _empty_glcm_features(self) -> Dict[str, float]:
        """Return empty GLCM features"""
        return {
            'contrast': 0.0,
            'dissimilarity': 0.0,
            'homogeneity': 0.0,
            'energy': 0.0,
            'correlation': 0.0,
            'ASM': 0.0
        }
    
    def extract_lbp_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract Local Binary Pattern texture histogram.
        
        Useful for detecting fine-grained texture patterns.
        """
        if image is None or mask is None:
            return np.zeros(10)
        
        try:
            from skimage.feature import local_binary_pattern
        except ImportError:
            print("⚠️  skimage not available - returning empty LBP features")
            return np.zeros(10)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Extract histogram only from masked region
        lbp_masked = lbp[mask > 0]
        
        if len(lbp_masked) == 0:
            return np.zeros(n_points + 2)
        
        # Histogram of LBP
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp_masked.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float) / (hist.sum() + 1e-10)  # Normalize
        
        return hist
    
    def extract_color_histograms(self, image: np.ndarray, mask: np.ndarray, 
                                  bins: int = 32) -> Dict[str, np.ndarray]:
        """
        Extract color histograms in multiple color spaces.
        
        Color spaces: RGB, HSV, LAB, YCrCb
        """
        if image is None or mask is None:
            return {}
        
        histograms = {}
        
        conversions = {
            'RGB': None,
            'HSV': cv2.COLOR_BGR2HSV,
            'LAB': cv2.COLOR_BGR2LAB,
            'YCrCb': cv2.COLOR_BGR2YCrCb
        }
        
        for space_name, conversion in conversions.items():
            if conversion is None:
                converted = image
            else:
                converted = cv2.cvtColor(image, conversion)
            
            hist = []
            for i in range(3):
                h = cv2.calcHist([converted], [i], mask, [bins], [0, 256])
                h = h.flatten() / (h.sum() + 1e-10)
                hist.append(h)
            
            histograms[space_name] = np.concatenate(hist)
        
        return histograms
    
    def calculate_polarization_ratio(self, 
                                     parallel: Optional[np.ndarray],
                                     cross: Optional[np.ndarray],
                                     mask: np.ndarray) -> float:
        """
        Calculate ratio of parallel to cross-polarized light intensity.
        
        High ratio indicates more surface reflection (keratinization).
        """
        if parallel is None or cross is None or mask is None:
            return 1.0  # Neutral
        
        # Extract masked regions
        parallel_masked = parallel[mask > 0]
        cross_masked = cross[mask > 0]
        
        if len(parallel_masked) == 0:
            return 1.0
        
        parallel_mean = np.mean(parallel_masked)
        cross_mean = np.mean(cross_masked)
        
        return parallel_mean / (cross_mean + 1e-10)
    
    def analyze_uv_fluorescence(self, uv_image: Optional[np.ndarray], 
                                mask: np.ndarray) -> Dict[str, float]:
        """
        Analyze UV fluorescence patterns within lesion.
        
        Abnormal cells often show different fluorescence.
        """
        if uv_image is None or mask is None or np.sum(mask) == 0:
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'max_intensity': 0.0,
                'spatial_heterogeneity': 0.0,
                'fluorescence_ratio': 0.0
            }
        
        gray_uv = cv2.cvtColor(uv_image, cv2.COLOR_BGR2GRAY) if len(uv_image.shape) == 3 else uv_image
        
        # Extract masked region
        uv_masked = gray_uv[mask > 0]
        
        if len(uv_masked) == 0:
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'max_intensity': 0.0,
                'spatial_heterogeneity': 0.0,
                'fluorescence_ratio': 0.0
            }
        
        mean_int = float(np.mean(uv_masked))
        std_int = float(np.std(uv_masked))
        max_int = float(np.max(uv_masked))
        
        # Spatial heterogeneity
        spatial_het = std_int / (mean_int + 1e-10)
        
        # Compare to surrounding normal skin
        dilated_mask = cv2.dilate(mask, np.ones((30, 30), np.uint8))
        surrounding_mask = dilated_mask - mask
        surrounding_pixels = gray_uv[surrounding_mask > 0]
        
        if len(surrounding_pixels) > 0:
            fluorescence_ratio = mean_int / (np.mean(surrounding_pixels) + 1e-10)
        else:
            fluorescence_ratio = 1.0
        
        return {
            'mean_intensity': mean_int,
            'std_intensity': std_int,
            'max_intensity': max_int,
            'spatial_heterogeneity': float(spatial_het),
            'fluorescence_ratio': float(fluorescence_ratio)
        }
    
    def analyze_melanin_distribution(self, multispectral: Dict[str, np.ndarray],
                                     mask: np.ndarray) -> Dict[str, float]:
        """
        Analyze melanin distribution from multispectral data.
        """
        if 'melanin_index' not in multispectral or mask is None or np.sum(mask) == 0:
            return {
                'mean_melanin': 0.0,
                'melanin_heterogeneity': 0.0,
                'high_melanin_fraction': 0.0,
                'melanin_asymmetry': 0.0
            }
        
        melanin = multispectral['melanin_index']
        melanin_masked = melanin[mask > 0]
        
        if len(melanin_masked) == 0:
            return {
                'mean_melanin': 0.0,
                'melanin_heterogeneity': 0.0,
                'high_melanin_fraction': 0.0,
                'melanin_asymmetry': 0.0
            }
        
        mean_mel = float(np.nanmean(melanin_masked))
        het_mel = float(np.nanstd(melanin_masked))
        
        # Fraction of high melanin pixels
        threshold = np.nanpercentile(melanin_masked, 75)
        high_frac = float(np.sum(melanin_masked > threshold) / len(melanin_masked))
        
        # Melanin asymmetry (spatial distribution)
        h, w = melanin.shape
        left_melanin = melanin[:, :w//2][mask[:, :w//2] > 0]
        right_melanin = melanin[:, w//2:][mask[:, w//2:] > 0]
        
        if len(left_melanin) > 0 and len(right_melanin) > 0:
            asymmetry = abs(np.nanmean(left_melanin) - np.nanmean(right_melanin))
        else:
            asymmetry = 0.0
        
        return {
            'mean_melanin': mean_mel,
            'melanin_heterogeneity': het_mel,
            'high_melanin_fraction': high_frac,
            'melanin_asymmetry': float(asymmetry)
        }
    
    def analyze_vascular_patterns(self, multispectral: Dict[str, np.ndarray],
                                  mask: np.ndarray) -> Dict[str, float]:
        """
        Analyze vascular patterns from red channel emphasis.
        
        SCC often shows increased vascularity.
        """
        if 'erythema_index' not in multispectral or mask is None or np.sum(mask) == 0:
            return {
                'mean_vascularity': 0.0,
                'vascular_heterogeneity': 0.0,
                'high_vascular_fraction': 0.0,
                'vascular_pattern_score': 0.0
            }
        
        erythema = multispectral['erythema_index']
        erythema_masked = erythema[mask > 0]
        
        if len(erythema_masked) == 0:
            return {
                'mean_vascularity': 0.0,
                'vascular_heterogeneity': 0.0,
                'high_vascular_fraction': 0.0,
                'vascular_pattern_score': 0.0
            }
        
        mean_vasc = float(np.nanmean(erythema_masked))
        het_vasc = float(np.nanstd(erythema_masked))
        
        # High vascularity fraction
        threshold = np.nanpercentile(erythema_masked, 75)
        high_frac = float(np.sum(erythema_masked > threshold) / len(erythema_masked))
        
        # Pattern score based on distribution characteristics
        skewness = float(np.mean((erythema_masked - mean_vasc) ** 3) / (het_vasc ** 3 + 1e-10))
        pattern_score = min(1.0, abs(skewness) * 0.5 + het_vasc * 0.5)
        
        return {
            'mean_vascularity': mean_vasc,
            'vascular_heterogeneity': het_vasc,
            'high_vascular_fraction': high_frac,
            'vascular_pattern_score': pattern_score
        }
    
    def calculate_seven_point_checklist(self, asymmetry: float, border: float,
                                        color: float, glcm: Dict, 
                                        vascular: Dict) -> float:
        """
        Calculate 7-point checklist score for melanoma screening.
        
        Major criteria (2 points each): atypical pigment network, blue-white veil, 
                                        atypical vascular pattern
        Minor criteria (1 point each): irregular streaks, irregular pigmentation,
                                       irregular dots/globules, regression structures
        
        Score >= 3 suggests high risk
        """
        score = 0.0
        
        # Major criteria (approximated from extracted features)
        if color > 0.5:  # Atypical pigmentation pattern
            score += 2.0
        if vascular.get('vascular_pattern_score', 0) > 0.5:  # Atypical vascular
            score += 2.0
        if glcm.get('contrast', 0) > 50:  # Structural changes
            score += 2.0
        
        # Minor criteria
        if asymmetry > 0.3:  # Irregular shape
            score += 1.0
        if border > 0.3:  # Irregular border
            score += 1.0
        if glcm.get('homogeneity', 1) < 0.5:  # Irregular texture
            score += 1.0
        if color > 0.3:  # Color variation
            score += 1.0
        
        return min(7.0, score)
    
    def calculate_three_point_checklist(self, asymmetry: float, color: float,
                                        vascular: Dict) -> float:
        """
        Calculate 3-point checklist for quick screening.
        
        1. Asymmetry of color and structure
        2. Atypical pigment network
        3. Blue-white structures
        
        Any 1 positive = suspect lesion
        """
        score = 0.0
        
        if asymmetry > 0.4:
            score += 1.0
        if color > 0.4:
            score += 1.0
        if vascular.get('mean_vascularity', 0) > 1.5:
            score += 1.0
        
        return score
    
    def generate_feature_vector(self, features: LesionFeatures) -> np.ndarray:
        """
        Generate flat feature vector for ML input.
        
        Returns:
            1D numpy array with all numerical features
        """
        vector = []
        
        # ABCDE
        vector.extend([
            features.asymmetry_score,
            features.border_irregularity,
            features.color_variation,
            features.diameter_mm
        ])
        
        # GLCM
        for key in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            vector.append(features.glcm_features.get(key, 0.0))
        
        # LBP histogram
        vector.extend(features.lbp_histogram.tolist())
        
        # Polarization
        vector.append(features.polarization_ratio)
        
        # UV features
        for key in ['mean_intensity', 'std_intensity', 'spatial_heterogeneity', 'fluorescence_ratio']:
            vector.append(features.uv_features.get(key, 0.0))
        
        # Melanin features
        for key in ['mean_melanin', 'melanin_heterogeneity', 'high_melanin_fraction', 'melanin_asymmetry']:
            vector.append(features.melanin_features.get(key, 0.0))
        
        # Vascular features
        for key in ['mean_vascularity', 'vascular_heterogeneity', 'high_vascular_fraction', 'vascular_pattern_score']:
            vector.append(features.vascular_features.get(key, 0.0))
        
        # Clinical scores
        vector.extend([
            features.seven_point_score,
            features.three_point_score
        ])
        
        return np.array(vector, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in the feature vector"""
        names = [
            'asymmetry_score', 'border_irregularity', 'color_variation', 'diameter_mm',
            'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 
            'glcm_energy', 'glcm_correlation', 'glcm_ASM'
        ]
        
        # LBP features (variable length based on radius)
        names.extend([f'lbp_{i}' for i in range(26)])  # 8*3 + 2 for uniform LBP
        
        names.extend([
            'polarization_ratio',
            'uv_mean_intensity', 'uv_std_intensity', 'uv_spatial_heterogeneity', 'uv_fluorescence_ratio',
            'melanin_mean', 'melanin_heterogeneity', 'melanin_high_fraction', 'melanin_asymmetry',
            'vascular_mean', 'vascular_heterogeneity', 'vascular_high_fraction', 'vascular_pattern_score',
            'seven_point_score', 'three_point_score'
        ])
        
        return names

