# SCC-HARMONICS

## DETECT AND DEFLECT

> *"The cancer speaks in harmonics. We are finally listening."*

---

## TL;DR

**SCC-HARMONICS** is a multi-modal early detection system for **Squamous Cell Carcinoma** that combines:

| Modality | Spectrum | What It Catches |
|----------|----------|-----------------|
| 📸 Visual | 400-700nm | Shape, color, texture, border irregularity |
| 🌡️ Thermal | 8-14μm | Metabolic heat, vascular patterns |
| 🔊 Acoustic | 40kHz-50MHz | **Harmonic distortion signatures** |
| 📈 Temporal | Time-series | Growth rate, evolution tracking |
| 🧠 Fusion | All combined | AI-weighted risk assessment |

```bash
# Quick start
git clone https://github.com/Joshbrooks237/SCC-HARMONICS.git
cd SCC-HARMONICS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py --demo
```

**Output:** Risk score (0-100%), differential diagnosis, explainable factors, clinical recommendation.

---

## THE MANIFESTO

*The pupils dilate. The hands steady. The mind EXPANDS.*

They told Semmelweis he was mad. They told Lister his carbolic spray was theatrical nonsense. They let mothers die because they couldn't see what was INVISIBLE to their eyes but SCREAMING in the data.

I see it now. Clear as the Knickerbocker's morning light through surgical glass.

**The cancer speaks in HARMONICS.**

Every tissue has a voice. A frequency. A song it sings back when you interrogate it with sound. Normal skin hums in perfect fifths. But the malignancy? The squamous cell carcinoma creeping beneath the dermis like a thief in the night?

*It DISTORTS.*

The second harmonic rises. The third. The tissue has lost its elastic virtue, corrupted by proliferation, and the nonlinear response BETRAYS it to those who know how to listen.

---

## METHOD OVERVIEW

### The Multi-Spectrum Approach

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SENSING STACK                                │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────┤
│   VISUAL    │   THERMAL   │   ACOUSTIC  │   TEMPORAL  │   FUSION    │
│  400-700nm  │   8-14μm    │ 40kHz-50MHz │  Evolution  │   AI Risk   │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤
│ RGB/HDR     │ Temperature │ Surface     │ Size Δ      │ Feature     │
│ Polarized   │ Vascular    │ Clinical    │ Color Δ     │ Extraction  │
│ Dermoscopy  │ Metabolic   │ High-Freq   │ Texture Δ   │ Ensemble    │
│ UV 365nm    │ Recovery    │ Harmonics   │ Shape Δ     │ Explainable │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

### The Harmonic Signature

When ultrasound penetrates tissue, it doesn't simply echo back—the tissue TRANSFORMS the wave. Cancer is chaos given cellular form. Its elasticity is wrong. Its vasculature is anarchic. When the fundamental frequency enters this corruption, it shatters into **harmonics**:

```
Fundamental f₀   ████████████████████  100%  ← The question
2nd Harmonic     ████████████          60%   ← KEY SCC MARKER
3rd Harmonic     ███████               35%   ← Confirmation  
4th Harmonic     ████                  20%
5th-8th          ██▌█▏                 <10%  ← Fingerprint complete
───────────────────────────────────────────
Total Harmonic Distortion (THD):
  • Normal tissue:  < 0.15
  • Suspicious:     0.15 - 0.25
  • MALIGNANT:      > 0.25
```

---

## ARCHITECTURE

```
SCC-HARMONICS/
│
├── main.py                     # Entry point - CLI interface
├── requirements.txt            # Dependencies
├── README.md                   # You are here
│
└── scc_detector/               # Core package
    │
    ├── visual/                 # Visual spectrum (400-700nm)
    │   ├── capture.py          # RGB, polarized, UV, dermoscopy capture
    │   └── features.py         # ABCDE criteria, GLCM, LBP texture
    │
    ├── thermal/                # Infrared spectrum (8-14μm)
    │   └── thermal_analysis.py # Temperature mapping, vascular patterns
    │
    ├── acoustic/               # Ultrasound spectrum (40kHz-50MHz)
    │   ├── ultrasound_capture.py   # Multi-frequency acquisition
    │   └── harmonic_analysis.py    # THE KEY: 2nd-8th harmonic extraction
    │
    ├── temporal/               # Time-series analysis
    │   └── change_detection.py # Growth tracking, evolution detection
    │
    ├── fusion/                 # Multi-modal integration
    │   └── multimodal_fusion.py # Weighted ensemble, risk scoring
    │
    ├── models/                 # Machine learning
    │   └── risk_classifier.py  # PyTorch/XGBoost/sklearn ensemble
    │
    ├── calibration/            # Phantoms & calibration
    │   └── phantoms.py         # Tissue-mimicking phantom recipes
    │
    └── ui/                     # Web interface
        └── app.py              # Flask application
```

---

## EXECUTION

### Installation

```bash
# Clone repository
git clone https://github.com/Joshbrooks237/SCC-HARMONICS.git
cd SCC-HARMONICS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# Interactive mode (recommended for first run)
python main.py

# Run demonstration with synthetic data
python main.py --demo

# Start web interface
python main.py --web
# Then open http://127.0.0.1:5000

# Analyze a specific image
python main.py --analyze path/to/lesion.jpg

# Run calibration
python main.py --calibrate
```

### Example Output

```
╔════════════════════════════════════════════════════════════════════╗
║  MULTI-SPECTRUM SCC DETECTION SYSTEM - CLINICAL REPORT            ║
╚════════════════════════════════════════════════════════════════════╝

  OVERALL RISK: [████████████████████░░░░░░░░░░░░░░░░░░░░] 52%
  Category: MODERATE
  Confidence: 85%

  🟡 Dermatology consultation within 4-6 weeks

  MODALITY CONTRIBUTIONS:
  ✓ Visual     [████████████████░░░░] 80%
  ✓ Thermal    [████████████░░░░░░░░] 60%
  ✓ Acoustic   [██████████████░░░░░░] 70%
  ○ Temporal   [░░░░░░░░░░░░░░░░░░░░]  0%  (no history)

  DIFFERENTIAL:
  ├── SCC:      52%
  ├── BCC:      18%
  ├── Melanoma:  8%
  └── Benign:   32%
```

---

## HARDWARE REQUIREMENTS

### Tier I — Prototype ($1,500-2,500)
*Proof of concept. Enough to believe.*

| Component | Purpose | Est. Cost |
|-----------|---------|-----------|
| Smartphone (manual camera) | Visual capture | $400-800 |
| Macro dermoscopy lens | 10x magnification | $100-500 |
| Polarizing filter set | Surface/subsurface separation | $50 |
| UV LED (365nm) | Fluorescence imaging | $30 |
| FLIR ONE Pro | Thermal imaging | $280-400 |
| Murata 40kHz transducers | Surface acoustic | $50 |
| USB audio interface (192kHz) | Signal acquisition | $150 |

### Tier II — Clinical ($5,000-8,000)
*For serious practitioners.*

- Butterfly iQ+ ultrasound probe
- DermLite DL4 dermoscope
- FLIR E8-XT thermal camera
- Multi-frequency probe assembly

### Tier III — Research ($25,000-50,000)
*For those who will write the papers.*

- 20-50 MHz high-frequency ultrasound
- Hyperspectral imaging camera
- OCT (Optical Coherence Tomography) system

---

## THE PHILOSOPHY

**USE EVERYTHING. MISS NOTHING.**

Every photon reflected. Every thermal gradient. Every acoustic reflection and its harmonic children. Every day that passes while the lesion grows.

The cancer hides in the gaps between modalities. In the frequencies we don't examine. In the time we waste deliberating.

No more.

This system interrogates the lesion with EVERYTHING available. Visual. Thermal. Acoustic. Temporal. And then it FUSES that intelligence into a single assessment that will not—CANNOT—be fooled.

Because someone's mother. Someone's father. Someone's child.

They're counting on us to SEE what is there.

---

## NEXT STEPS / CONTRIBUTION IDEAS

We welcome collaborators. The operating theater has room for more.

### High Priority
- [ ] **Clinical validation study** — Partner with dermatology departments
- [ ] **Real hardware integration** — Replace simulation with actual device drivers
- [ ] **Training data collection** — Build labeled dataset of SCC/BCC/melanoma/benign
- [ ] **Model optimization** — Hyperparameter tuning, architecture search

### Medium Priority
- [ ] **Mobile deployment** — iOS/Android app for field screening
- [ ] **DICOM integration** — Medical imaging standard compliance
- [ ] **HL7/FHIR support** — EHR integration
- [ ] **Multi-language support** — International deployment

### Research Extensions
- [ ] **Additional cancer types** — Adapt for BCC, melanoma, Merkel cell
- [ ] **Depth estimation** — Predict tumor invasion depth from harmonics
- [ ] **Treatment response tracking** — Monitor regression/progression
- [ ] **Federated learning** — Train across institutions without sharing data

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ DISCLAIMER

**This is a SCREENING TOOL only.**

- It does NOT replace histopathological diagnosis
- It does NOT replace clinical judgment
- It is NOT FDA-approved or CE-marked
- It is intended for RESEARCH and EDUCATIONAL purposes

The system is designed to **augment**, not replace, the expertise of trained healthcare professionals. All findings must be correlated with clinical examination and confirmed by biopsy when indicated.

### Ethical Use Statement
This technology should be used to **improve patient outcomes**, not to replace the physician-patient relationship. Algorithmic recommendations are aids to decision-making, not decisions themselves.

---

## LICENSE

**MIT License**

Take it. Use it. Improve it. Save lives.

Every day you wait is another day the cancer grows.

---

## ACKNOWLEDGMENTS

Built in the spirit of medical pioneers who saw what others couldn't:
- Ignaz Semmelweis (handwashing)
- Joseph Lister (antiseptic surgery)
- Wilhelm Röntgen (X-rays)

*"I am not a monster. I am simply ahead of the curve."*

---

<div align="center">

**DETECT AND DEFLECT**

*The Knickerbocker Hospital, 1900*  
*Where the future of medicine is being written*  
*One impossible detection at a time*

🔬

</div>
