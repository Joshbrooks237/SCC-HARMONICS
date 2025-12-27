#!/usr/bin/env python3
"""
SCC Multi-Spectrum Early Detection System
=========================================

A comprehensive, multi-modal screening system for early detection of 
Squamous Cell Carcinoma using visual, thermal, and acoustic sensing.

Usage:
    python main.py                    # Run interactive CLI demo
    python main.py --web              # Start web interface
    python main.py --analyze IMAGE    # Analyze a specific image
    python main.py --calibrate        # Run calibration procedure
    python main.py --demo             # Run full demonstration

Philosophy: Use Everything. Miss Nothing.
"""

import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print the system banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗ ██████╗ ██████╗    ██████╗ ███████╗████████╗███████╗ ██████╗████████╗║
║   ██╔════╝██╔════╝██╔════╝    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝║
║   ███████╗██║     ██║         ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ║
║   ╚════██║██║     ██║         ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ║
║   ███████║╚██████╗╚██████╗    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ║
║   ╚══════╝ ╚═════╝ ╚═════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   ║
║                                                                               ║
║             Multi-Spectrum Early Detection System v0.1.0                      ║
║                                                                               ║
║   Visual | Thermal | Acoustic | Temporal | AI Fusion                         ║
║   ───────────────────────────────────────────────────────────────             ║
║   Philosophy: Use Everything. Miss Nothing.                                   ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_demo():
    """Run a full demonstration of the system"""
    print("\n" + "=" * 70)
    print("RUNNING FULL SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Import all modules
    print("\n[1/8] Loading modules...")
    
    try:
        from scc_detector.visual import MultiSpectrumVisualCapture, VisualFeatureExtractor
        from scc_detector.thermal import ThermalImagingSystem
        from scc_detector.acoustic import UltrasoundHarmonicAnalyzer, UltrasoundHardwareInterface
        from scc_detector.temporal import TemporalChangeDetector
        from scc_detector.fusion import MultiModalFusionEngine
        from scc_detector.calibration import TissuePhantom, CalibrationSystem
        print("✓ All modules loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        print("  Please ensure all dependencies are installed: pip install -r requirements.txt")
        return
    
    # Initialize systems
    print("\n[2/8] Initializing capture systems...")
    visual_capture = MultiSpectrumVisualCapture(simulation_mode=True)
    thermal_system = ThermalImagingSystem(simulation_mode=True)
    ultrasound_hardware = UltrasoundHardwareInterface(simulation_mode=True)
    harmonic_analyzer = UltrasoundHarmonicAnalyzer()
    temporal_detector = TemporalChangeDetector()
    fusion_engine = MultiModalFusionEngine()
    visual_extractor = VisualFeatureExtractor()
    
    # Run capture sequence
    print("\n[3/8] Running visual spectrum capture...")
    visual_data = visual_capture.guided_capture_sequence(
        patient_id="DEMO001",
        lesion_id="L001",
        body_location="left forearm"
    )
    print(f"✓ Visual capture complete")
    print(f"  - RGB image: {visual_data.rgb_standard.shape}")
    print(f"  - Quality score: {visual_data.metadata.get('overall_quality', 'N/A')}")
    
    # Extract visual features
    print("\n[4/8] Extracting visual features...")
    visual_features = visual_extractor.extract_all_features(visual_data)
    visual_vector = visual_extractor.generate_feature_vector(visual_features)
    print(f"✓ Visual features extracted")
    print(f"  - Asymmetry: {visual_features.asymmetry_score:.3f}")
    print(f"  - Border irregularity: {visual_features.border_irregularity:.3f}")
    print(f"  - Color variation: {visual_features.color_variation:.3f}")
    print(f"  - Diameter: {visual_features.diameter_mm:.1f} mm")
    print(f"  - 7-point score: {visual_features.seven_point_score:.1f}")
    
    # Thermal capture
    print("\n[5/8] Capturing thermal signature...")
    thermal_data = thermal_system.capture_thermal_snapshot()
    thermal_features = thermal_system.extract_thermal_features(thermal_data)
    thermal_vector = thermal_system.generate_thermal_feature_vector(thermal_features)
    print(f"✓ Thermal capture complete")
    print(f"  - Mean temperature: {thermal_features.mean_temp:.1f}°C")
    print(f"  - ΔT from surrounding: {thermal_features.delta_T:.2f}°C")
    print(f"  - Vascular index: {thermal_features.vascular_index:.3f}")
    
    # Ultrasound harmonic analysis
    print("\n[6/8] Running ultrasound harmonic analysis...")
    captures = ultrasound_hardware.capture_full_spectrum()
    acoustic_features = harmonic_analyzer.analyze_multi_frequency(captures)
    acoustic_vector = harmonic_analyzer.generate_feature_vector(acoustic_features)
    print(f"✓ Harmonic analysis complete")
    print(f"  - Frequencies analyzed: {len(captures)}")
    print(f"  - Mean THD: {acoustic_features.profile.mean_thd:.3f}")
    print(f"  - SCC harmonic score: {acoustic_features.scc_harmonic_score:.3f}")
    print(f"  - Depth penetration: {acoustic_features.depth_penetration:.2f}")
    
    # Print harmonic analysis report
    print("\n" + harmonic_analyzer.explain_harmonic_findings(acoustic_features))
    
    # Multi-modal fusion
    print("\n[7/8] Running multi-modal fusion...")
    fused_features = fusion_engine.fuse_features(
        visual_features=visual_vector,
        thermal_features=thermal_vector,
        acoustic_features=acoustic_vector,
        temporal_features=None  # No historical data in demo
    )
    print(f"✓ Features fused")
    print(f"  - Visual weight: {fused_features.modality_weights.get('visual', 0):.2%}")
    print(f"  - Thermal weight: {fused_features.modality_weights.get('thermal', 0):.2%}")
    print(f"  - Acoustic weight: {fused_features.modality_weights.get('acoustic', 0):.2%}")
    print(f"  - Fused vector dimension: {len(fused_features.fused_vector)}")
    
    # Risk assessment
    print("\n[8/8] Performing risk assessment...")
    assessment = fusion_engine.assess_risk(fused_features)
    
    # Generate and print report
    report = fusion_engine.generate_comprehensive_report(
        assessment,
        fused_features,
        patient_info={'id': 'DEMO001', 'location': 'left forearm'}
    )
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return assessment


def run_calibration():
    """Run system calibration"""
    print("\n[CALIBRATION MODE]")
    
    from scc_detector.calibration import TissuePhantom, CalibrationSystem
    
    # Show phantom recipes
    print("\n--- Tissue Phantom Recipes ---")
    print(TissuePhantom.get_recipe('skin_agar'))
    
    # Run calibration
    print("\n--- Running System Calibration ---")
    cal_system = CalibrationSystem()
    
    # Create synthetic calibration targets
    phantom = TissuePhantom('skin_dermis')
    visual_target = phantom.simulate_visual_response()
    thermal_target = phantom.simulate_thermal_response()
    acoustic_target = phantom.simulate_acoustic_response()
    
    result = cal_system.run_full_calibration(
        visual_capture=visual_target,
        thermal_capture=thermal_target,
        acoustic_signal=acoustic_target
    )
    
    print("\n" + cal_system.get_calibration_report())
    
    return result


def analyze_image(image_path: str):
    """Analyze a specific image"""
    import cv2
    import numpy as np
    
    print(f"\n[ANALYZING IMAGE: {image_path}]")
    
    if not os.path.exists(image_path):
        print(f"✗ Error: File not found: {image_path}")
        return None
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Error: Could not read image: {image_path}")
        return None
    
    print(f"✓ Image loaded: {image.shape}")
    
    # Import modules
    from scc_detector.visual import VisualFeatureExtractor
    from scc_detector.visual.capture import VisualCapture
    from scc_detector.fusion import MultiModalFusionEngine
    
    # Create visual capture object
    visual_data = VisualCapture(
        rgb_standard=image,
        timestamp=datetime.now(),
        patient_id="CLI",
        lesion_id="L001"
    )
    
    # Extract features
    print("\nExtracting features...")
    extractor = VisualFeatureExtractor()
    features = extractor.extract_all_features(visual_data)
    feature_vector = extractor.generate_feature_vector(features)
    
    print(f"\n--- VISUAL ANALYSIS ---")
    print(f"Asymmetry Score:      {features.asymmetry_score:.3f}")
    print(f"Border Irregularity:  {features.border_irregularity:.3f}")
    print(f"Color Variation:      {features.color_variation:.3f}")
    print(f"Diameter (estimated): {features.diameter_mm:.1f} mm")
    print(f"7-Point Score:        {features.seven_point_score:.1f}/7")
    print(f"3-Point Score:        {features.three_point_score:.1f}/3")
    
    # Texture analysis
    print(f"\n--- TEXTURE ANALYSIS ---")
    print(f"GLCM Contrast:        {features.glcm_features.get('contrast', 0):.2f}")
    print(f"GLCM Homogeneity:     {features.glcm_features.get('homogeneity', 0):.3f}")
    print(f"GLCM Energy:          {features.glcm_features.get('energy', 0):.4f}")
    
    # Run through fusion for risk estimate
    fusion = MultiModalFusionEngine()
    fused = fusion.fuse_features(visual_features=feature_vector)
    assessment = fusion.assess_risk(fused)
    
    print(f"\n--- RISK ASSESSMENT ---")
    print(f"Risk Score:           {assessment.risk_score:.1%}")
    print(f"Risk Category:        {assessment.risk_category.upper()}")
    print(f"Confidence:           {assessment.confidence:.1%}")
    print(f"\nRecommendation: {assessment.recommendation}")
    
    return assessment


def run_web_interface(host='127.0.0.1', port=5000):
    """Start the web interface"""
    print("\n[STARTING WEB INTERFACE]")
    
    try:
        from scc_detector.ui import run_app
        run_app(host=host, port=port, debug=True)
    except ImportError as e:
        print(f"✗ Error: Could not start web interface: {e}")
        print("  Please install Flask: pip install flask")


def run_interactive():
    """Run interactive CLI mode"""
    print_banner()
    
    print("\nWelcome to the SCC Multi-Spectrum Detection System!")
    print("This system combines visual, thermal, and acoustic analysis")
    print("to detect squamous cell carcinoma at the earliest possible stage.")
    
    while True:
        print("\n" + "-" * 50)
        print("OPTIONS:")
        print("  1. Run full demonstration")
        print("  2. Start web interface")
        print("  3. Analyze an image file")
        print("  4. Run calibration")
        print("  5. View system information")
        print("  6. Exit")
        print("-" * 50)
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            run_web_interface()
        elif choice == '3':
            path = input("Enter image path: ").strip()
            if path:
                analyze_image(path)
        elif choice == '4':
            run_calibration()
        elif choice == '5':
            print_system_info()
        elif choice == '6':
            print("\nThank you for using SCC Detector. Stay vigilant!")
            break
        else:
            print("Invalid choice. Please try again.")


def print_system_info():
    """Print system information"""
    info = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    SYSTEM INFORMATION                         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  VISUAL SPECTRUM (400-700nm)                                  ║
    ║  ├─ Standard RGB with HDR                                     ║
    ║  ├─ Cross & Parallel Polarization                             ║
    ║  ├─ Dermoscopy (10x magnification)                            ║
    ║  ├─ UV Fluorescence (365nm)                                   ║
    ║  └─ Multispectral Indices                                     ║
    ║                                                               ║
    ║  THERMAL SPECTRUM (8-14μm)                                    ║
    ║  ├─ Surface Temperature Mapping                               ║
    ║  ├─ Vascular Pattern Detection                                ║
    ║  ├─ Metabolic Heat Signatures                                 ║
    ║  └─ Dynamic Recovery Analysis                                 ║
    ║                                                               ║
    ║  ACOUSTIC SPECTRUM (40kHz - 50MHz)                            ║
    ║  ├─ Surface Acoustic (40-200 kHz)                             ║
    ║  ├─ Clinical Ultrasound (5-15 MHz)                            ║
    ║  ├─ High-Frequency Research (20-50 MHz)                       ║
    ║  └─ Full Harmonic Analysis (2nd-8th harmonics)                ║
    ║                                                               ║
    ║  TEMPORAL ANALYSIS                                            ║
    ║  ├─ Size Progression Tracking                                 ║
    ║  ├─ Color Evolution Monitoring                                ║
    ║  ├─ Texture Change Detection                                  ║
    ║  └─ Growth Rate Modeling                                      ║
    ║                                                               ║
    ║  AI FUSION ENGINE                                             ║
    ║  ├─ Multi-Modal Feature Extraction                            ║
    ║  ├─ Deep Learning Integration                                 ║
    ║  ├─ Explainable Risk Scoring                                  ║
    ║  └─ Real-Time Risk Stratification                             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(info)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SCC Multi-Spectrum Early Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                    # Interactive mode
  python main.py --demo             # Run demonstration
  python main.py --web              # Start web server
  python main.py --analyze img.jpg  # Analyze an image
  python main.py --calibrate        # Run calibration
        '''
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run full system demonstration')
    parser.add_argument('--web', action='store_true',
                       help='Start web interface')
    parser.add_argument('--analyze', type=str, metavar='IMAGE',
                       help='Analyze a specific image file')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run system calibration')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Web server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Web server port (default: 5000)')
    
    args = parser.parse_args()
    
    if args.demo:
        print_banner()
        run_demo()
    elif args.web:
        print_banner()
        run_web_interface(host=args.host, port=args.port)
    elif args.analyze:
        print_banner()
        analyze_image(args.analyze)
    elif args.calibrate:
        print_banner()
        run_calibration()
    else:
        # Interactive mode
        run_interactive()


if __name__ == '__main__':
    main()

