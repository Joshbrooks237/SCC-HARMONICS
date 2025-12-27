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

---
"The operating theater is no place for the timid.
 Step forward, or step aside."
                                    - In the spirit of The Knick
---
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - Gather your instruments. The procedure begins.
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import sys
import os
from datetime import datetime

# The path to enlightenment must be explicit
# Only the bold may proceed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """
    Print the system banner.
    
    First impressions matter. Even in medicine.
    Especially in medicine.
    """
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
    """
    Run a full demonstration of the system.
    
    The curtain rises. The lights focus.
    This is what we've been building toward.
    """
    print("\n" + "=" * 70)
    print("RUNNING FULL SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Summon the instruments
    # Each module is a blade. Together, they are a symphony.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/8] Loading modules...")
    
    try:
        # The visual eye - sees what light reveals
        from scc_detector.visual import MultiSpectrumVisualCapture, VisualFeatureExtractor
        # The thermal eye - sees what heat betrays
        from scc_detector.thermal import ThermalImagingSystem
        # The acoustic ear - hears what echoes confess
        from scc_detector.acoustic import UltrasoundHarmonicAnalyzer, UltrasoundHardwareInterface
        # The temporal memory - remembers what time changes
        from scc_detector.temporal import TemporalChangeDetector
        # The fusion mind - synthesizes what none alone could grasp
        from scc_detector.fusion import MultiModalFusionEngine
        # The calibration truth - grounds us in reality
        from scc_detector.calibration import TissuePhantom, CalibrationSystem
        print("✓ All modules loaded successfully")
    except ImportError as e:
        # A surgeon without instruments is merely a witness
        print(f"✗ Failed to import modules: {e}")
        print("  Please ensure all dependencies are installed: pip install -r requirements.txt")
        return
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Initialize the operating theater
    # Every instrument in its place. Every sensor at the ready.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/8] Initializing capture systems...")
    
    # In simulation mode, we create our own reality
    # Reality will come soon enough
    visual_capture = MultiSpectrumVisualCapture(simulation_mode=True)
    thermal_system = ThermalImagingSystem(simulation_mode=True)
    ultrasound_hardware = UltrasoundHardwareInterface(simulation_mode=True)
    harmonic_analyzer = UltrasoundHarmonicAnalyzer()
    temporal_detector = TemporalChangeDetector()
    fusion_engine = MultiModalFusionEngine()
    visual_extractor = VisualFeatureExtractor()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: The visual interrogation
    # What does the lesion show to those who truly look?
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/8] Running visual spectrum capture...")
    
    # The patient presents. We observe with every wavelength at our disposal.
    visual_data = visual_capture.guided_capture_sequence(
        patient_id="DEMO001",
        lesion_id="L001",
        body_location="left forearm"
    )
    print(f"✓ Visual capture complete")
    print(f"  - RGB image: {visual_data.rgb_standard.shape}")
    print(f"  - Quality score: {visual_data.metadata.get('overall_quality', 'N/A')}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: Extract the visual evidence
    # ABCDE - the alphabet of suspicion
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/8] Extracting visual features...")
    
    # The dermoscope reveals what the naked eye dismisses
    visual_features = visual_extractor.extract_all_features(visual_data)
    visual_vector = visual_extractor.generate_feature_vector(visual_features)
    
    print(f"✓ Visual features extracted")
    print(f"  - Asymmetry: {visual_features.asymmetry_score:.3f}")          # A - is it balanced?
    print(f"  - Border irregularity: {visual_features.border_irregularity:.3f}")  # B - are edges smooth?
    print(f"  - Color variation: {visual_features.color_variation:.3f}")    # C - is color uniform?
    print(f"  - Diameter: {visual_features.diameter_mm:.1f} mm")            # D - size matters
    print(f"  - 7-point score: {visual_features.seven_point_score:.1f}")    # The clinical verdict
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: The thermal confession
    # Cancer burns hotter. It cannot hide its metabolic hunger.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[5/8] Capturing thermal signature...")
    
    # Heat is truth. The tumor's appetite betrays it.
    thermal_data = thermal_system.capture_thermal_snapshot()
    thermal_features = thermal_system.extract_thermal_features(thermal_data)
    thermal_vector = thermal_system.generate_thermal_feature_vector(thermal_features)
    
    print(f"✓ Thermal capture complete")
    print(f"  - Mean temperature: {thermal_features.mean_temp:.1f}°C")
    print(f"  - ΔT from surrounding: {thermal_features.delta_T:.2f}°C")     # The fever of proliferation
    print(f"  - Vascular index: {thermal_features.vascular_index:.3f}")     # Blood feeds the beast
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 6: The harmonic interrogation
    # THIS IS WHERE THE MAGIC HAPPENS.
    # The cancer SPEAKS in harmonics. We have learned to LISTEN.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[6/8] Running ultrasound harmonic analysis...")
    
    # From 40 kHz to 50 MHz - we sweep the entire acoustic spectrum
    # Each frequency a question. Each harmonic an answer.
    captures = ultrasound_hardware.capture_full_spectrum()
    
    # The analyzer extracts the harmonic fingerprint
    # Normal tissue hums in tune. Cancer DISTORTS.
    acoustic_features = harmonic_analyzer.analyze_multi_frequency(captures)
    acoustic_vector = harmonic_analyzer.generate_feature_vector(acoustic_features)
    
    print(f"✓ Harmonic analysis complete")
    print(f"  - Frequencies analyzed: {len(captures)}")
    print(f"  - Mean THD: {acoustic_features.profile.mean_thd:.3f}")        # Total Harmonic Distortion
    print(f"  - SCC harmonic score: {acoustic_features.scc_harmonic_score:.3f}")  # The acoustic verdict
    print(f"  - Depth penetration: {acoustic_features.depth_penetration:.2f}")    # How deep does it go?
    
    # The harmonic report - for those who need to understand WHY
    print("\n" + harmonic_analyzer.explain_harmonic_findings(acoustic_features))
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 7: The fusion synthesis
    # One mind to rule them all. One verdict from many voices.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[7/8] Running multi-modal fusion...")
    
    # Each modality is a witness. Fusion is the jury.
    fused_features = fusion_engine.fuse_features(
        visual_features=visual_vector,
        thermal_features=thermal_vector,
        acoustic_features=acoustic_vector,
        temporal_features=None  # No historical data in demo - time is yet to tell
    )
    
    print(f"✓ Features fused")
    print(f"  - Visual weight: {fused_features.modality_weights.get('visual', 0):.2%}")
    print(f"  - Thermal weight: {fused_features.modality_weights.get('thermal', 0):.2%}")
    print(f"  - Acoustic weight: {fused_features.modality_weights.get('acoustic', 0):.2%}")
    print(f"  - Fused vector dimension: {len(fused_features.fused_vector)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 8: The verdict
    # All evidence considered. All modalities weighed. 
    # What is our assessment?
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[8/8] Performing risk assessment...")
    
    # The moment of truth. What have we found?
    assessment = fusion_engine.assess_risk(fused_features)
    
    # The comprehensive report - transparent, explainable, actionable
    report = fusion_engine.generate_comprehensive_report(
        assessment,
        fused_features,
        patient_info={'id': 'DEMO001', 'location': 'left forearm'}
    )
    print("\n" + report)
    
    # ─────────────────────────────────────────────────────────────────────────
    # CURTAIN CALL
    # The procedure is complete. The patient's fate is now in better hands.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return assessment


def run_calibration():
    """
    Run system calibration.
    
    Trust nothing. Verify everything.
    A surgeon who does not calibrate his instruments 
    calibrates only his arrogance.
    """
    print("\n[CALIBRATION MODE]")
    
    from scc_detector.calibration import TissuePhantom, CalibrationSystem
    
    # Show phantom recipes - the art of creating synthetic truth
    print("\n--- Tissue Phantom Recipes ---")
    print(TissuePhantom.get_recipe('skin_agar'))
    
    # Run calibration - align the instruments with reality
    print("\n--- Running System Calibration ---")
    cal_system = CalibrationSystem()
    
    # Create synthetic calibration targets
    # We must know what truth looks like before we can detect lies
    phantom = TissuePhantom('skin_dermis')
    visual_target = phantom.simulate_visual_response()
    thermal_target = phantom.simulate_thermal_response()
    acoustic_target = phantom.simulate_acoustic_response()
    
    result = cal_system.run_full_calibration(
        visual_capture=visual_target,
        thermal_capture=thermal_target,
        acoustic_signal=acoustic_target
    )
    
    # The calibration verdict
    print("\n" + cal_system.get_calibration_report())
    
    return result


def analyze_image(image_path: str):
    """
    Analyze a specific image.
    
    One image. One chance. 
    Let us not waste it.
    """
    import cv2
    import numpy as np
    
    print(f"\n[ANALYZING IMAGE: {image_path}]")
    
    # First, verify the specimen exists
    if not os.path.exists(image_path):
        print(f"✗ Error: File not found: {image_path}")
        return None
    
    # Load the evidence
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Error: Could not read image: {image_path}")
        return None
    
    print(f"✓ Image loaded: {image.shape}")
    
    # Summon the specialists
    from scc_detector.visual import VisualFeatureExtractor
    from scc_detector.visual.capture import VisualCapture
    from scc_detector.fusion import MultiModalFusionEngine
    
    # Prepare the specimen for analysis
    visual_data = VisualCapture(
        rgb_standard=image,
        timestamp=datetime.now(),
        patient_id="CLI",
        lesion_id="L001"
    )
    
    # Extract every feature the image will yield
    print("\nExtracting features...")
    extractor = VisualFeatureExtractor()
    features = extractor.extract_all_features(visual_data)
    feature_vector = extractor.generate_feature_vector(features)
    
    # Report: The visual confession
    print(f"\n--- VISUAL ANALYSIS ---")
    print(f"Asymmetry Score:      {features.asymmetry_score:.3f}")
    print(f"Border Irregularity:  {features.border_irregularity:.3f}")
    print(f"Color Variation:      {features.color_variation:.3f}")
    print(f"Diameter (estimated): {features.diameter_mm:.1f} mm")
    print(f"7-Point Score:        {features.seven_point_score:.1f}/7")
    print(f"3-Point Score:        {features.three_point_score:.1f}/3")
    
    # Texture tells tales the eye cannot read directly
    print(f"\n--- TEXTURE ANALYSIS ---")
    print(f"GLCM Contrast:        {features.glcm_features.get('contrast', 0):.2f}")
    print(f"GLCM Homogeneity:     {features.glcm_features.get('homogeneity', 0):.3f}")
    print(f"GLCM Energy:          {features.glcm_features.get('energy', 0):.4f}")
    
    # Even with only visual data, we render a verdict
    fusion = MultiModalFusionEngine()
    fused = fusion.fuse_features(visual_features=feature_vector)
    assessment = fusion.assess_risk(fused)
    
    # The final assessment
    print(f"\n--- RISK ASSESSMENT ---")
    print(f"Risk Score:           {assessment.risk_score:.1%}")
    print(f"Risk Category:        {assessment.risk_category.upper()}")
    print(f"Confidence:           {assessment.confidence:.1%}")
    print(f"\nRecommendation: {assessment.recommendation}")
    
    return assessment


def run_web_interface(host='127.0.0.1', port=5000):
    """
    Start the web interface.
    
    For those who prefer their surgery with a GUI.
    We do not judge. We only heal.
    """
    print("\n[STARTING WEB INTERFACE]")
    
    try:
        # The web unfurls. The interface awakens.
        from scc_detector.ui import run_app
        run_app(host=host, port=port, debug=True)
    except ImportError as e:
        print(f"✗ Error: Could not start web interface: {e}")
        print("  Please install Flask: pip install flask")


def run_interactive():
    """
    Run interactive CLI mode.
    
    The command line: where true operators operate.
    """
    print_banner()
    
    print("\nWelcome to the SCC Multi-Spectrum Detection System!")
    print("This system combines visual, thermal, and acoustic analysis")
    print("to detect squamous cell carcinoma at the earliest possible stage.")
    
    # The eternal loop of inquiry
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
            run_demo()          # The full experience
        elif choice == '2':
            run_web_interface() # For the visually inclined
        elif choice == '3':
            path = input("Enter image path: ").strip()
            if path:
                analyze_image(path)  # Single-specimen analysis
        elif choice == '4':
            run_calibration()   # Trust but verify
        elif choice == '5':
            print_system_info() # The specifications
        elif choice == '6':
            # The procedure is complete
            print("\nThank you for using SCC Detector. Stay vigilant!")
            break
        else:
            print("Invalid choice. Please try again.")


def print_system_info():
    """
    Print system information.
    
    Know your instruments. Know their capabilities.
    Know their limits.
    """
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
    """
    Main entry point.
    
    Where every journey through the system begins.
    Choose your path. Face what you find.
    """
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
    
    # The arguments of fate
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
    
    # Dispatch based on the surgeon's command
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
        # No specific command - enter the interactive realm
        run_interactive()


# ═══════════════════════════════════════════════════════════════════════════════
# THE THRESHOLD
# Cross it, and the procedure begins.
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    main()
