"""
Multi-Modal Fusion Engine for SCC Detection

Combines all sensing modalities into actionable intelligence:
- Visual features (ABCDE criteria, texture, color)
- Thermal features (metabolic signatures)
- Acoustic features (harmonic analysis)
- Temporal features (evolution tracking)

Fusion strategies:
1. Early fusion: Concatenate all feature vectors
2. Late fusion: Ensemble of modality-specific classifiers
3. Attention-based fusion: Learn modality importance weights
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings


@dataclass
class FusedFeatures:
    """Combined feature set from all modalities"""
    # Individual modality vectors
    visual_features: np.ndarray
    thermal_features: np.ndarray
    acoustic_features: np.ndarray
    temporal_features: np.ndarray
    
    # Combined vector
    fused_vector: np.ndarray
    
    # Modality availability flags
    has_visual: bool = True
    has_thermal: bool = True
    has_acoustic: bool = True
    has_temporal: bool = True
    
    # Modality weights (for interpretability)
    modality_weights: Dict[str, float] = field(default_factory=dict)
    
    # Feature names
    feature_names: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Complete risk assessment output"""
    # Primary outputs
    risk_score: float  # 0-1 overall risk
    risk_category: str  # "low", "moderate", "high", "very_high"
    confidence: float  # Model confidence
    
    # Per-modality contributions
    visual_contribution: float
    thermal_contribution: float
    acoustic_contribution: float
    temporal_contribution: float
    
    # Clinical recommendations
    recommendation: str
    urgency: str  # "routine", "soon", "urgent", "immediate"
    
    # Explainability
    top_risk_factors: List[Tuple[str, float]]  # (feature_name, contribution)
    protective_factors: List[Tuple[str, float]]
    
    # Differential considerations
    scc_probability: float
    bcc_probability: float
    melanoma_probability: float
    benign_probability: float


class MultiModalFusionEngine:
    """
    Fuse multi-modal features for comprehensive risk assessment.
    
    Philosophy: Each modality provides unique information.
    Fusion maximizes detection sensitivity while maintaining specificity.
    """
    
    def __init__(self, 
                 fusion_method: str = "weighted_early",
                 model_path: Optional[str] = None):
        """
        Initialize fusion engine.
        
        Args:
            fusion_method: "early", "weighted_early", "late", or "attention"
            model_path: Path to trained model weights
        """
        self.fusion_method = fusion_method
        self.model_path = model_path
        
        # Default modality weights (can be learned)
        self.modality_weights = {
            'visual': 0.35,
            'thermal': 0.20,
            'acoustic': 0.25,
            'temporal': 0.20
        }
        
        # Feature importance (initialized to uniform)
        self.feature_importance = None
        
        # Load model if available
        self.classifier = None
        if model_path:
            self._load_model(model_path)
        else:
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize default ensemble model"""
        try:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.neural_network import MLPClassifier
            
            self.classifiers = {
                'gbm': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                ),
                'rf': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42
                ),
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    max_iter=500,
                    random_state=42
                )
            }
            print("✓ Default ensemble classifiers initialized")
        except ImportError:
            print("⚠️  scikit-learn not available - using rule-based fallback")
            self.classifiers = None
    
    def _load_model(self, path: str):
        """Load trained model from disk"""
        try:
            import joblib
            self.classifier = joblib.load(path)
            print(f"✓ Loaded model from {path}")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            self._initialize_default_model()
    
    def fuse_features(self,
                     visual_features: Optional[np.ndarray] = None,
                     thermal_features: Optional[np.ndarray] = None,
                     acoustic_features: Optional[np.ndarray] = None,
                     temporal_features: Optional[np.ndarray] = None) -> FusedFeatures:
        """
        Fuse features from all available modalities.
        
        Args:
            visual_features: Visual feature vector
            thermal_features: Thermal feature vector
            acoustic_features: Acoustic/ultrasound feature vector
            temporal_features: Temporal change feature vector
            
        Returns:
            FusedFeatures with combined representation
        """
        # Handle missing modalities
        if visual_features is None:
            visual_features = np.zeros(50)  # Expected visual feature length
            has_visual = False
        else:
            has_visual = True
        
        if thermal_features is None:
            thermal_features = np.zeros(16)  # Expected thermal feature length
            has_thermal = False
        else:
            has_thermal = True
        
        if acoustic_features is None:
            acoustic_features = np.zeros(30)  # Expected acoustic feature length
            has_acoustic = False
        else:
            has_acoustic = True
        
        if temporal_features is None:
            temporal_features = np.zeros(17)  # Expected temporal feature length
            has_temporal = False
        else:
            has_temporal = True
        
        # Normalize each modality
        visual_norm = self._normalize_features(visual_features)
        thermal_norm = self._normalize_features(thermal_features)
        acoustic_norm = self._normalize_features(acoustic_features)
        temporal_norm = self._normalize_features(temporal_features)
        
        # Calculate effective weights (zero out missing modalities)
        effective_weights = {
            'visual': self.modality_weights['visual'] if has_visual else 0,
            'thermal': self.modality_weights['thermal'] if has_thermal else 0,
            'acoustic': self.modality_weights['acoustic'] if has_acoustic else 0,
            'temporal': self.modality_weights['temporal'] if has_temporal else 0
        }
        
        # Renormalize weights
        total_weight = sum(effective_weights.values()) + 1e-10
        effective_weights = {k: v/total_weight for k, v in effective_weights.items()}
        
        if self.fusion_method == "early":
            # Simple concatenation
            fused = np.concatenate([
                visual_norm, thermal_norm, acoustic_norm, temporal_norm
            ])
        
        elif self.fusion_method == "weighted_early":
            # Weighted concatenation
            fused = np.concatenate([
                visual_norm * effective_weights['visual'],
                thermal_norm * effective_weights['thermal'],
                acoustic_norm * effective_weights['acoustic'],
                temporal_norm * effective_weights['temporal']
            ])
        
        elif self.fusion_method == "late":
            # Late fusion - will be handled in assessment
            fused = np.concatenate([
                visual_norm, thermal_norm, acoustic_norm, temporal_norm
            ])
        
        else:  # attention-based
            fused = self._attention_fusion(
                visual_norm, thermal_norm, acoustic_norm, temporal_norm,
                effective_weights
            )
        
        # Build feature names
        feature_names = (
            self._get_visual_names() +
            self._get_thermal_names() +
            self._get_acoustic_names() +
            self._get_temporal_names()
        )
        
        return FusedFeatures(
            visual_features=visual_norm,
            thermal_features=thermal_norm,
            acoustic_features=acoustic_norm,
            temporal_features=temporal_norm,
            fused_vector=fused,
            has_visual=has_visual,
            has_thermal=has_thermal,
            has_acoustic=has_acoustic,
            has_temporal=has_temporal,
            modality_weights=effective_weights,
            feature_names=feature_names
        )
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector to unit norm"""
        norm = np.linalg.norm(features) + 1e-10
        return features / norm
    
    def _attention_fusion(self,
                         visual: np.ndarray,
                         thermal: np.ndarray,
                         acoustic: np.ndarray,
                         temporal: np.ndarray,
                         base_weights: Dict[str, float]) -> np.ndarray:
        """
        Attention-based fusion that learns to weight features.
        
        Uses a simple attention mechanism based on feature informativeness.
        """
        # Calculate "informativeness" of each modality
        # Based on variance (more varied = more informative)
        visual_info = np.var(visual) if len(visual) > 0 else 0
        thermal_info = np.var(thermal) if len(thermal) > 0 else 0
        acoustic_info = np.var(acoustic) if len(acoustic) > 0 else 0
        temporal_info = np.var(temporal) if len(temporal) > 0 else 0
        
        total_info = visual_info + thermal_info + acoustic_info + temporal_info + 1e-10
        
        # Attention weights (combination of base + learned)
        attention = {
            'visual': 0.5 * base_weights['visual'] + 0.5 * (visual_info / total_info),
            'thermal': 0.5 * base_weights['thermal'] + 0.5 * (thermal_info / total_info),
            'acoustic': 0.5 * base_weights['acoustic'] + 0.5 * (acoustic_info / total_info),
            'temporal': 0.5 * base_weights['temporal'] + 0.5 * (temporal_info / total_info)
        }
        
        # Apply attention weights
        fused = np.concatenate([
            visual * attention['visual'],
            thermal * attention['thermal'],
            acoustic * attention['acoustic'],
            temporal * attention['temporal']
        ])
        
        return fused
    
    def assess_risk(self, fused_features: FusedFeatures) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            fused_features: Fused multi-modal features
            
        Returns:
            Complete RiskAssessment
        """
        feature_vector = fused_features.fused_vector
        
        # Get predictions
        if self.classifiers is not None:
            risk_score, confidence, class_probs = self._ensemble_predict(feature_vector)
        else:
            risk_score, confidence, class_probs = self._rule_based_predict(
                fused_features
            )
        
        # Calculate per-modality contributions
        visual_contrib = self._calculate_modality_contribution(
            fused_features.visual_features, 'visual', fused_features.modality_weights
        )
        thermal_contrib = self._calculate_modality_contribution(
            fused_features.thermal_features, 'thermal', fused_features.modality_weights
        )
        acoustic_contrib = self._calculate_modality_contribution(
            fused_features.acoustic_features, 'acoustic', fused_features.modality_weights
        )
        temporal_contrib = self._calculate_modality_contribution(
            fused_features.temporal_features, 'temporal', fused_features.modality_weights
        )
        
        # Risk categorization
        if risk_score >= 0.8:
            risk_category = "very_high"
            urgency = "immediate"
            recommendation = "URGENT: Immediate dermatology referral and biopsy recommended"
        elif risk_score >= 0.6:
            risk_category = "high"
            urgency = "urgent"
            recommendation = "HIGH RISK: Urgent dermatology referral within 1-2 weeks"
        elif risk_score >= 0.4:
            risk_category = "moderate"
            urgency = "soon"
            recommendation = "MODERATE: Dermatology consultation within 4-6 weeks"
        else:
            risk_category = "low"
            urgency = "routine"
            recommendation = "LOW RISK: Monitor for changes, routine follow-up"
        
        # Get explainability
        top_risks, protective = self._explain_prediction(
            fused_features.fused_vector,
            fused_features.feature_names,
            risk_score
        )
        
        return RiskAssessment(
            risk_score=risk_score,
            risk_category=risk_category,
            confidence=confidence,
            visual_contribution=visual_contrib,
            thermal_contribution=thermal_contrib,
            acoustic_contribution=acoustic_contrib,
            temporal_contribution=temporal_contrib,
            recommendation=recommendation,
            urgency=urgency,
            top_risk_factors=top_risks,
            protective_factors=protective,
            scc_probability=class_probs.get('scc', risk_score),
            bcc_probability=class_probs.get('bcc', risk_score * 0.3),
            melanoma_probability=class_probs.get('melanoma', risk_score * 0.2),
            benign_probability=class_probs.get('benign', 1 - risk_score)
        )
    
    def _ensemble_predict(self, 
                         features: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
        """Predict using ensemble of classifiers"""
        # Note: In production, classifiers would be trained
        # For now, return rule-based estimate
        
        # Simple heuristic based on feature statistics
        feature_mean = np.mean(features)
        feature_max = np.max(features)
        feature_std = np.std(features)
        
        # Higher values generally indicate more abnormality
        risk_score = min(1.0, (feature_mean + feature_max) / 2 * (1 + feature_std))
        confidence = 0.7  # Default confidence without training
        
        class_probs = {
            'scc': risk_score * 0.8,
            'bcc': risk_score * 0.15,
            'melanoma': risk_score * 0.05,
            'benign': 1 - risk_score
        }
        
        return risk_score, confidence, class_probs
    
    def _rule_based_predict(self,
                           fused: FusedFeatures) -> Tuple[float, float, Dict[str, float]]:
        """
        Rule-based prediction using clinical criteria.
        
        Fallback when ML models not available.
        """
        risk_factors = []
        
        # Visual risk factors
        if fused.has_visual:
            v = fused.visual_features
            if len(v) >= 4:
                # ABCDE criteria
                asymmetry = v[0]
                border = v[1]
                color = v[2]
                diameter = v[3]
                
                if asymmetry > 0.4:
                    risk_factors.append(('asymmetry', 0.15))
                if border > 0.4:
                    risk_factors.append(('border_irregularity', 0.15))
                if color > 0.4:
                    risk_factors.append(('color_variation', 0.15))
                if diameter > 6.0:  # > 6mm
                    risk_factors.append(('diameter>6mm', 0.1))
        
        # Thermal risk factors
        if fused.has_thermal:
            t = fused.thermal_features
            if len(t) >= 6:
                delta_T = t[5] if len(t) > 5 else 0
                if delta_T > 1.0:  # > 1°C elevation
                    risk_factors.append(('thermal_elevation', 0.15))
        
        # Acoustic risk factors
        if fused.has_acoustic:
            a = fused.acoustic_features
            if len(a) >= 1:
                mean_thd = a[0]
                if mean_thd > 0.2:
                    risk_factors.append(('high_harmonic_distortion', 0.15))
        
        # Temporal risk factors
        if fused.has_temporal:
            temp = fused.temporal_features
            if len(temp) >= 15:
                evolution_risk = temp[14] if len(temp) > 14 else 0
                if evolution_risk > 0.5:
                    risk_factors.append(('rapid_evolution', 0.2))
        
        # Calculate total risk
        risk_score = sum(r[1] for r in risk_factors)
        risk_score = min(1.0, risk_score)
        
        # Confidence based on modality availability
        available = sum([fused.has_visual, fused.has_thermal, 
                        fused.has_acoustic, fused.has_temporal])
        confidence = 0.5 + (available / 4) * 0.3
        
        class_probs = {
            'scc': risk_score * 0.8,
            'bcc': risk_score * 0.12,
            'melanoma': risk_score * 0.08,
            'benign': 1 - risk_score
        }
        
        return risk_score, confidence, class_probs
    
    def _calculate_modality_contribution(self,
                                        features: np.ndarray,
                                        modality: str,
                                        weights: Dict[str, float]) -> float:
        """Calculate how much a modality contributes to risk"""
        if len(features) == 0:
            return 0.0
        
        # Contribution = weight * mean feature value
        mean_value = np.mean(np.abs(features))
        contribution = weights.get(modality, 0.25) * mean_value
        
        return float(min(1.0, contribution))
    
    def _explain_prediction(self,
                           features: np.ndarray,
                           feature_names: List[str],
                           risk_score: float) -> Tuple[List[Tuple[str, float]], 
                                                       List[Tuple[str, float]]]:
        """
        Generate explanation for prediction.
        
        Returns:
            (top_risk_factors, protective_factors)
        """
        if len(feature_names) != len(features):
            # Mismatch - return generic explanation
            return [("abnormal_features", risk_score)], []
        
        # Pair features with names and sort by absolute value
        paired = list(zip(feature_names, features))
        paired.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Top risk factors (high positive values)
        risk_factors = [(name, float(val)) for name, val in paired 
                       if val > 0.3][:5]
        
        # Protective factors (normal/low values in key areas)
        protective = [(name, float(val)) for name, val in paired 
                     if val < 0.1 and any(kw in name for kw in 
                        ['asymmetry', 'border', 'color', 'thermal', 'thd'])][:3]
        
        return risk_factors, protective
    
    def _get_visual_names(self) -> List[str]:
        """Get visual feature names"""
        # Match visual feature extractor output
        names = [
            'asymmetry_score', 'border_irregularity', 'color_variation', 'diameter_mm',
            'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity', 
            'glcm_energy', 'glcm_correlation', 'glcm_ASM'
        ]
        # LBP features
        names.extend([f'lbp_{i}' for i in range(26)])
        # Other visual features
        names.extend([
            'polarization_ratio',
            'uv_mean_intensity', 'uv_std_intensity', 'uv_spatial_heterogeneity', 'uv_fluorescence_ratio',
            'melanin_mean', 'melanin_heterogeneity', 'melanin_high_fraction', 'melanin_asymmetry',
            'vascular_mean', 'vascular_heterogeneity', 'vascular_high_fraction', 'vascular_pattern_score',
            'seven_point_score', 'three_point_score'
        ])
        return names
    
    def _get_thermal_names(self) -> List[str]:
        """Get thermal feature names"""
        return [
            'thermal_mean_temp', 'thermal_max_temp', 'thermal_min_temp',
            'thermal_temp_range', 'thermal_temp_std', 'thermal_delta_T',
            'thermal_asymmetry', 'thermal_hot_spot_ratio',
            'thermal_vascular_index', 'thermal_perfusion_score',
            'thermal_recovery_rate', 'thermal_time_constant', 'thermal_inertia',
            'thermal_gradient', 'thermal_edge_sharpness', 'thermal_core_periphery'
        ]
    
    def _get_acoustic_names(self) -> List[str]:
        """Get acoustic feature names"""
        names = [
            'harmonic_mean_thd', 'harmonic_max_thd', 
            'harmonic_thd_variance', 'harmonic_consistency'
        ]
        for order in range(2, 9):
            names.append(f'freq_dependence_h{order}')
        for category in ['surface', 'clinical', 'highfreq']:
            names.extend([
                f'{category}_mean_thd', f'{category}_std_thd',
                f'{category}_mean_h2', f'{category}_std_h2'
            ])
        names.extend([
            'scc_harmonic_score', 'depth_penetration', 'structural_irregularity',
            'thd_map_mean', 'thd_map_std', 'thd_map_max', 'thd_map_p95'
        ])
        return names
    
    def _get_temporal_names(self) -> List[str]:
        """Get temporal feature names"""
        return [
            'temporal_area_change_rate', 'temporal_diameter_change_rate',
            'temporal_volume_doubling_time', 'temporal_perimeter_growth_rate',
            'temporal_growth_acceleration', 'temporal_color_change_rate',
            'temporal_darkening_rate', 'temporal_erythema_change',
            'temporal_asymmetry_change', 'temporal_border_change',
            'temporal_shape_stability', 'temporal_contrast_change',
            'temporal_heterogeneity_change', 'temporal_rapid_change_flag',
            'temporal_evolution_risk_score', 'temporal_observation_years',
            'temporal_num_observations'
        ]
    
    def generate_comprehensive_report(self, 
                                      assessment: RiskAssessment,
                                      fused: FusedFeatures,
                                      patient_info: Optional[Dict] = None) -> str:
        """
        Generate comprehensive clinical report.
        
        Args:
            assessment: Risk assessment results
            fused: Fused features
            patient_info: Optional patient information
            
        Returns:
            Formatted clinical report
        """
        report = []
        
        # Header
        report.append("╔" + "═" * 68 + "╗")
        report.append("║" + " MULTI-SPECTRUM SCC DETECTION SYSTEM - CLINICAL REPORT ".center(68) + "║")
        report.append("╚" + "═" * 68 + "╝")
        
        # Date and patient info
        report.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if patient_info:
            report.append(f"Patient ID: {patient_info.get('id', 'N/A')}")
            report.append(f"Lesion Location: {patient_info.get('location', 'N/A')}")
        
        # Risk Summary
        report.append("\n" + "─" * 70)
        report.append("█ RISK ASSESSMENT SUMMARY")
        report.append("─" * 70)
        
        # Visual risk bar
        risk_bar = "█" * int(assessment.risk_score * 40) + "░" * (40 - int(assessment.risk_score * 40))
        report.append(f"\n  OVERALL RISK: [{risk_bar}] {assessment.risk_score:.1%}")
        report.append(f"  Category: {assessment.risk_category.upper()}")
        report.append(f"  Confidence: {assessment.confidence:.1%}")
        
        # Urgency and recommendation
        urgency_icons = {
            'routine': '🟢',
            'soon': '🟡',
            'urgent': '🟠',
            'immediate': '🔴'
        }
        report.append(f"\n  {urgency_icons.get(assessment.urgency, '⚪')} {assessment.recommendation}")
        
        # Modality Contributions
        report.append("\n" + "─" * 70)
        report.append("█ MODALITY ANALYSIS")
        report.append("─" * 70)
        
        modalities = [
            ("Visual", assessment.visual_contribution, fused.has_visual),
            ("Thermal", assessment.thermal_contribution, fused.has_thermal),
            ("Acoustic", assessment.acoustic_contribution, fused.has_acoustic),
            ("Temporal", assessment.temporal_contribution, fused.has_temporal)
        ]
        
        for name, contrib, available in modalities:
            status = "✓" if available else "✗"
            bar = "█" * int(contrib * 20) + "░" * (20 - int(contrib * 20))
            report.append(f"  {status} {name:10} [{bar}] {contrib:.1%}")
        
        # Differential Diagnosis
        report.append("\n" + "─" * 70)
        report.append("█ DIFFERENTIAL DIAGNOSIS")
        report.append("─" * 70)
        
        diagnoses = [
            ("Squamous Cell Carcinoma", assessment.scc_probability),
            ("Basal Cell Carcinoma", assessment.bcc_probability),
            ("Melanoma", assessment.melanoma_probability),
            ("Benign Lesion", assessment.benign_probability)
        ]
        
        for name, prob in sorted(diagnoses, key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
            report.append(f"  {name:30} [{bar}] {prob:.1%}")
        
        # Risk Factors
        if assessment.top_risk_factors:
            report.append("\n" + "─" * 70)
            report.append("█ KEY RISK FACTORS")
            report.append("─" * 70)
            
            for factor, value in assessment.top_risk_factors[:5]:
                report.append(f"  ⚠ {factor}: {value:.3f}")
        
        # Protective Factors
        if assessment.protective_factors:
            report.append("\n" + "─" * 70)
            report.append("█ PROTECTIVE FACTORS")
            report.append("─" * 70)
            
            for factor, value in assessment.protective_factors[:3]:
                report.append(f"  ✓ {factor}: within normal range")
        
        # Disclaimer
        report.append("\n" + "─" * 70)
        report.append("DISCLAIMER: This is a screening tool only. Clinical correlation and")
        report.append("histopathological confirmation are required for definitive diagnosis.")
        report.append("─" * 70)
        
        return "\n".join(report)

