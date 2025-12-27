"""
Web Application for SCC Multi-Spectrum Detection System

Clinical workflow interface:
1. Patient registration
2. Guided capture workflow
3. Real-time analysis
4. Risk assessment display
5. Report generation
6. Temporal tracking
"""

import os
import sys
import json
import base64
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import cv2


def create_app():
    """Create and configure the Flask application"""
    try:
        from flask import Flask, render_template_string, request, jsonify, send_file
    except ImportError:
        print("⚠️  Flask not installed. Run: pip install flask")
        return None
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'scc-detection-dev-key'
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
    
    # Initialize components
    from ..visual import MultiSpectrumVisualCapture, VisualFeatureExtractor
    from ..thermal import ThermalImagingSystem
    from ..acoustic import UltrasoundHarmonicAnalyzer, UltrasoundHardwareInterface
    from ..temporal import TemporalChangeDetector
    from ..fusion import MultiModalFusionEngine
    
    # Create instances (simulation mode for now)
    visual_capture = MultiSpectrumVisualCapture(simulation_mode=True)
    visual_extractor = VisualFeatureExtractor()
    thermal_system = ThermalImagingSystem(simulation_mode=True)
    ultrasound_hardware = UltrasoundHardwareInterface(simulation_mode=True)
    harmonic_analyzer = UltrasoundHarmonicAnalyzer()
    temporal_detector = TemporalChangeDetector()
    fusion_engine = MultiModalFusionEngine()
    
    # HTML template with embedded CSS and JS
    MAIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SCC Multi-Spectrum Detection System</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent-cyan: #00d4ff;
            --accent-magenta: #ff00aa;
            --accent-green: #00ff88;
            --accent-orange: #ff6b00;
            --accent-red: #ff3366;
            --text-primary: #e8e8e8;
            --text-secondary: #888899;
            --border-color: #2a2a3a;
            --gradient-main: linear-gradient(135deg, #00d4ff 0%, #ff00aa 100%);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Space Grotesk', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            background-image: 
                radial-gradient(ellipse at 20% 80%, rgba(0, 212, 255, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(255, 0, 170, 0.05) 0%, transparent 50%);
        }
        
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .logo-icon {
            width: 48px;
            height: 48px;
            background: var(--gradient-main);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
        }
        
        .logo-text h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: var(--gradient-main);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .logo-text p {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-tertiary);
            border-radius: 999px;
            font-size: 0.85rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 280px 1fr 320px;
            min-height: calc(100vh - 80px);
        }
        
        .sidebar {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            padding: 1.5rem;
        }
        
        .sidebar-section {
            margin-bottom: 2rem;
        }
        
        .sidebar-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }
        
        .nav-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 0.5rem;
        }
        
        .nav-item:hover {
            background: var(--bg-tertiary);
        }
        
        .nav-item.active {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid var(--accent-cyan);
        }
        
        .nav-icon {
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .content {
            padding: 2rem;
            overflow-y: auto;
        }
        
        .content-header {
            margin-bottom: 2rem;
        }
        
        .content-header h2 {
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }
        
        .content-header p {
            color: var(--text-secondary);
        }
        
        .capture-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }
        
        .capture-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s;
        }
        
        .capture-card:hover {
            border-color: var(--accent-cyan);
            transform: translateY(-2px);
        }
        
        .capture-preview {
            aspect-ratio: 4/3;
            background: var(--bg-tertiary);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        
        .capture-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .capture-placeholder {
            color: var(--text-secondary);
            text-align: center;
        }
        
        .capture-placeholder .icon {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .capture-info {
            padding: 1rem;
        }
        
        .capture-info h3 {
            font-size: 1rem;
            margin-bottom: 0.25rem;
        }
        
        .capture-info p {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .modality-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .modality-visual { background: rgba(0, 212, 255, 0.2); color: var(--accent-cyan); }
        .modality-thermal { background: rgba(255, 107, 0, 0.2); color: var(--accent-orange); }
        .modality-acoustic { background: rgba(0, 255, 136, 0.2); color: var(--accent-green); }
        .modality-temporal { background: rgba(255, 0, 170, 0.2); color: var(--accent-magenta); }
        
        .panel {
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            padding: 1.5rem;
        }
        
        .panel-section {
            margin-bottom: 2rem;
        }
        
        .panel-title {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .risk-meter {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .risk-score {
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .risk-low { color: var(--accent-green); }
        .risk-moderate { color: var(--accent-orange); }
        .risk-high { color: var(--accent-red); }
        
        .risk-bar {
            height: 8px;
            background: var(--bg-primary);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }
        
        .risk-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .risk-label {
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .modality-status {
            display: grid;
            gap: 0.5rem;
        }
        
        .modality-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            background: var(--bg-tertiary);
            border-radius: 6px;
        }
        
        .status-check {
            color: var(--accent-green);
        }
        
        .status-pending {
            color: var(--text-secondary);
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
        }
        
        .btn-primary {
            background: var(--gradient-main);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
        }
        
        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        
        .btn-secondary:hover {
            border-color: var(--accent-cyan);
        }
        
        .btn-block {
            width: 100%;
        }
        
        .upload-zone {
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-zone:hover {
            border-color: var(--accent-cyan);
            background: rgba(0, 212, 255, 0.05);
        }
        
        .upload-zone input {
            display: none;
        }
        
        .report-preview {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        
        .finding-item {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.75rem;
            background: var(--bg-tertiary);
            border-radius: 6px;
            margin-bottom: 0.5rem;
        }
        
        .finding-icon {
            font-size: 1.25rem;
        }
        
        .finding-content h4 {
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
        }
        
        .finding-content p {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }
        
        @media (max-width: 1200px) {
            .main-container {
                grid-template-columns: 1fr;
            }
            .sidebar, .panel {
                display: none;
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">
            <div class="logo-icon">🔬</div>
            <div class="logo-text">
                <h1>SCC Detector</h1>
                <p>Multi-Spectrum Early Detection System</p>
            </div>
        </div>
        <div class="status-indicator">
            <span class="status-dot"></span>
            <span>System Ready</span>
        </div>
    </header>
    
    <main class="main-container">
        <aside class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-title">Workflow</div>
                <div class="nav-item active" onclick="showSection('capture')">
                    <span class="nav-icon">📷</span>
                    <span>Capture</span>
                </div>
                <div class="nav-item" onclick="showSection('analysis')">
                    <span class="nav-icon">🔍</span>
                    <span>Analysis</span>
                </div>
                <div class="nav-item" onclick="showSection('history')">
                    <span class="nav-icon">📊</span>
                    <span>History</span>
                </div>
                <div class="nav-item" onclick="showSection('calibration')">
                    <span class="nav-icon">⚙️</span>
                    <span>Calibration</span>
                </div>
            </div>
            
            <div class="sidebar-section">
                <div class="sidebar-title">Patient</div>
                <div style="padding: 0.5rem;">
                    <input type="text" id="patientId" placeholder="Patient ID" 
                           style="width: 100%; padding: 0.5rem; background: var(--bg-tertiary); 
                                  border: 1px solid var(--border-color); border-radius: 6px; 
                                  color: var(--text-primary);">
                </div>
                <div style="padding: 0.5rem;">
                    <input type="text" id="lesionId" placeholder="Lesion ID" 
                           style="width: 100%; padding: 0.5rem; background: var(--bg-tertiary); 
                                  border: 1px solid var(--border-color); border-radius: 6px; 
                                  color: var(--text-primary);">
                </div>
            </div>
        </aside>
        
        <section class="content" id="captureSection">
            <div class="content-header">
                <h2>Multi-Modal Capture</h2>
                <p>Capture lesion data across all sensing modalities</p>
            </div>
            
            <div class="capture-grid">
                <div class="capture-card">
                    <div class="capture-preview" id="visualPreview">
                        <div class="capture-placeholder">
                            <div class="icon">📸</div>
                            <div>Click to upload image</div>
                        </div>
                    </div>
                    <div class="capture-info">
                        <span class="modality-badge modality-visual">Visual</span>
                        <h3>Dermoscopic Image</h3>
                        <p>RGB, polarized, UV capture</p>
                    </div>
                </div>
                
                <div class="capture-card">
                    <div class="capture-preview" id="thermalPreview">
                        <div class="capture-placeholder">
                            <div class="icon">🌡️</div>
                            <div>Thermal imaging</div>
                        </div>
                    </div>
                    <div class="capture-info">
                        <span class="modality-badge modality-thermal">Thermal</span>
                        <h3>Thermal Signature</h3>
                        <p>Infrared temperature mapping</p>
                    </div>
                </div>
                
                <div class="capture-card">
                    <div class="capture-preview" id="acousticPreview">
                        <div class="capture-placeholder">
                            <div class="icon">📡</div>
                            <div>Ultrasound scan</div>
                        </div>
                    </div>
                    <div class="capture-info">
                        <span class="modality-badge modality-acoustic">Acoustic</span>
                        <h3>Harmonic Analysis</h3>
                        <p>40 kHz - 50 MHz spectrum</p>
                    </div>
                </div>
                
                <div class="capture-card">
                    <div class="capture-preview" id="temporalPreview">
                        <div class="capture-placeholder">
                            <div class="icon">📈</div>
                            <div>Evolution tracking</div>
                        </div>
                    </div>
                    <div class="capture-info">
                        <span class="modality-badge modality-temporal">Temporal</span>
                        <h3>Change Detection</h3>
                        <p>Track lesion evolution</p>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2rem;">
                <div class="upload-zone" onclick="document.getElementById('imageUpload').click()">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📤</div>
                    <div>Drop files here or click to upload</div>
                    <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem;">
                        Supports: JPG, PNG, DICOM, TIFF
                    </div>
                    <input type="file" id="imageUpload" accept="image/*" onchange="handleUpload(this)">
                </div>
            </div>
            
            <div style="margin-top: 2rem; display: flex; gap: 1rem;">
                <button class="btn btn-primary" onclick="simulateCapture()">
                    🎯 Simulate Full Capture
                </button>
                <button class="btn btn-secondary" onclick="runAnalysis()">
                    🔬 Run Analysis
                </button>
            </div>
        </section>
        
        <aside class="panel">
            <div class="panel-section">
                <div class="panel-title">⚡ Risk Assessment</div>
                <div class="risk-meter">
                    <div class="risk-score risk-moderate" id="riskScore">--</div>
                    <div class="risk-bar">
                        <div class="risk-bar-fill" id="riskBar" 
                             style="width: 0%; background: var(--accent-green);"></div>
                    </div>
                    <div class="risk-label" id="riskLabel">Awaiting analysis...</div>
                </div>
            </div>
            
            <div class="panel-section">
                <div class="panel-title">📊 Modality Status</div>
                <div class="modality-status">
                    <div class="modality-row">
                        <span>Visual</span>
                        <span class="status-pending" id="statusVisual">○</span>
                    </div>
                    <div class="modality-row">
                        <span>Thermal</span>
                        <span class="status-pending" id="statusThermal">○</span>
                    </div>
                    <div class="modality-row">
                        <span>Acoustic</span>
                        <span class="status-pending" id="statusAcoustic">○</span>
                    </div>
                    <div class="modality-row">
                        <span>Temporal</span>
                        <span class="status-pending" id="statusTemporal">○</span>
                    </div>
                </div>
            </div>
            
            <div class="panel-section">
                <div class="panel-title">🔍 Key Findings</div>
                <div id="findingsContainer">
                    <div style="color: var(--text-secondary); text-align: center; padding: 1rem;">
                        Run analysis to see findings
                    </div>
                </div>
            </div>
            
            <div class="panel-section">
                <button class="btn btn-secondary btn-block" onclick="generateReport()">
                    📄 Generate Report
                </button>
            </div>
        </aside>
    </main>
    
    <script>
        let capturedData = {
            visual: null,
            thermal: null,
            acoustic: null,
            temporal: null
        };
        
        let analysisResults = null;
        
        function showSection(section) {
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            event.target.closest('.nav-item').classList.add('active');
        }
        
        function handleUpload(input) {
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    capturedData.visual = e.target.result;
                    document.getElementById('visualPreview').innerHTML = 
                        '<img src="' + e.target.result + '" alt="Captured">';
                    document.getElementById('statusVisual').textContent = '✓';
                    document.getElementById('statusVisual').className = 'status-check';
                };
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        async function simulateCapture() {
            // Simulate capturing data from all modalities
            const response = await fetch('/api/simulate-capture', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    patient_id: document.getElementById('patientId').value || 'TEST001',
                    lesion_id: document.getElementById('lesionId').value || 'L001'
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update previews
                if (data.visual_preview) {
                    document.getElementById('visualPreview').innerHTML = 
                        '<img src="data:image/png;base64,' + data.visual_preview + '" alt="Visual">';
                    document.getElementById('statusVisual').textContent = '✓';
                    document.getElementById('statusVisual').className = 'status-check';
                }
                
                if (data.thermal_preview) {
                    document.getElementById('thermalPreview').innerHTML = 
                        '<img src="data:image/png;base64,' + data.thermal_preview + '" alt="Thermal">';
                    document.getElementById('statusThermal').textContent = '✓';
                    document.getElementById('statusThermal').className = 'status-check';
                }
                
                document.getElementById('statusAcoustic').textContent = '✓';
                document.getElementById('statusAcoustic').className = 'status-check';
                
                document.getElementById('statusTemporal').textContent = '✓';
                document.getElementById('statusTemporal').className = 'status-check';
                
                capturedData = data.captured_data;
            }
        }
        
        async function runAnalysis() {
            document.getElementById('riskLabel').textContent = 'Analyzing...';
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    patient_id: document.getElementById('patientId').value || 'TEST001',
                    lesion_id: document.getElementById('lesionId').value || 'L001'
                })
            });
            
            const data = await response.json();
            analysisResults = data;
            
            if (data.success) {
                // Update risk display
                const riskScore = data.risk_score;
                const riskPercent = Math.round(riskScore * 100);
                
                document.getElementById('riskScore').textContent = riskPercent + '%';
                document.getElementById('riskBar').style.width = riskPercent + '%';
                
                let riskClass, riskColor;
                if (riskScore < 0.4) {
                    riskClass = 'risk-low';
                    riskColor = 'var(--accent-green)';
                    document.getElementById('riskLabel').textContent = 'Low Risk';
                } else if (riskScore < 0.7) {
                    riskClass = 'risk-moderate';
                    riskColor = 'var(--accent-orange)';
                    document.getElementById('riskLabel').textContent = 'Moderate Risk';
                } else {
                    riskClass = 'risk-high';
                    riskColor = 'var(--accent-red)';
                    document.getElementById('riskLabel').textContent = 'High Risk - Refer';
                }
                
                document.getElementById('riskScore').className = 'risk-score ' + riskClass;
                document.getElementById('riskBar').style.background = riskColor;
                
                // Update findings
                let findingsHtml = '';
                if (data.findings && data.findings.length > 0) {
                    data.findings.forEach(finding => {
                        let icon = '⚠️';
                        if (finding.severity === 'high') icon = '🔴';
                        else if (finding.severity === 'moderate') icon = '🟡';
                        else if (finding.severity === 'low') icon = '🟢';
                        
                        findingsHtml += `
                            <div class="finding-item">
                                <span class="finding-icon">${icon}</span>
                                <div class="finding-content">
                                    <h4>${finding.title}</h4>
                                    <p>${finding.description}</p>
                                </div>
                            </div>
                        `;
                    });
                } else {
                    findingsHtml = '<div style="color: var(--text-secondary); text-align: center; padding: 1rem;">No significant findings</div>';
                }
                document.getElementById('findingsContainer').innerHTML = findingsHtml;
            }
        }
        
        async function generateReport() {
            const response = await fetch('/api/generate-report', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    patient_id: document.getElementById('patientId').value || 'TEST001',
                    lesion_id: document.getElementById('lesionId').value || 'L001'
                })
            });
            
            const data = await response.json();
            
            if (data.success && data.report) {
                // Create download
                const blob = new Blob([data.report], { type: 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'scc_report_' + new Date().toISOString().slice(0,10) + '.txt';
                a.click();
            }
        }
    </script>
</body>
</html>
'''
    
    @app.route('/')
    def index():
        return render_template_string(MAIN_TEMPLATE)
    
    @app.route('/api/simulate-capture', methods=['POST'])
    def simulate_capture():
        """Simulate multi-modal capture"""
        data = request.get_json() or {}
        patient_id = data.get('patient_id', 'TEST001')
        lesion_id = data.get('lesion_id', 'L001')
        
        # Generate simulated captures
        visual_data = visual_capture._simulated_capture_sequence(patient_id, lesion_id, "forearm")
        thermal_data = thermal_system.capture_thermal_snapshot()
        
        # Encode images for preview
        _, visual_buffer = cv2.imencode('.png', visual_data.rgb_standard)
        visual_preview = base64.b64encode(visual_buffer).decode('utf-8')
        
        thermal_colormap = thermal_data.get_colormap_image()
        _, thermal_buffer = cv2.imencode('.png', thermal_colormap)
        thermal_preview = base64.b64encode(thermal_buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'visual_preview': visual_preview,
            'thermal_preview': thermal_preview,
            'captured_data': {
                'visual': True,
                'thermal': True,
                'acoustic': True,
                'temporal': True
            }
        })
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze():
        """Run full analysis pipeline"""
        data = request.get_json() or {}
        patient_id = data.get('patient_id', 'TEST001')
        lesion_id = data.get('lesion_id', 'L001')
        
        # Generate all data
        visual_data = visual_capture._simulated_capture_sequence(patient_id, lesion_id, "forearm")
        visual_features = visual_extractor.extract_all_features(visual_data)
        visual_vector = visual_extractor.generate_feature_vector(visual_features)
        
        thermal_data = thermal_system.capture_thermal_snapshot()
        thermal_features = thermal_system.extract_thermal_features(thermal_data)
        thermal_vector = thermal_system.generate_thermal_feature_vector(thermal_features)
        
        captures = ultrasound_hardware.capture_full_spectrum()
        acoustic_features = harmonic_analyzer.analyze_multi_frequency(captures)
        acoustic_vector = harmonic_analyzer.generate_feature_vector(acoustic_features)
        
        # Fuse and assess
        fused = fusion_engine.fuse_features(
            visual_features=visual_vector,
            thermal_features=thermal_vector,
            acoustic_features=acoustic_vector
        )
        
        assessment = fusion_engine.assess_risk(fused)
        
        # Generate findings
        findings = []
        
        if visual_features.asymmetry_score > 0.4:
            findings.append({
                'title': 'Asymmetric Shape',
                'description': f'Asymmetry score: {visual_features.asymmetry_score:.2f}',
                'severity': 'moderate'
            })
        
        if visual_features.border_irregularity > 0.4:
            findings.append({
                'title': 'Irregular Border',
                'description': f'Border irregularity: {visual_features.border_irregularity:.2f}',
                'severity': 'moderate'
            })
        
        if thermal_features.delta_T > 1.0:
            findings.append({
                'title': 'Elevated Temperature',
                'description': f'Temperature elevation: {thermal_features.delta_T:.1f}°C',
                'severity': 'moderate'
            })
        
        if acoustic_features.scc_harmonic_score > 0.5:
            findings.append({
                'title': 'Abnormal Harmonic Signature',
                'description': f'SCC harmonic score: {acoustic_features.scc_harmonic_score:.2f}',
                'severity': 'high' if acoustic_features.scc_harmonic_score > 0.7 else 'moderate'
            })
        
        return jsonify({
            'success': True,
            'risk_score': assessment.risk_score,
            'risk_category': assessment.risk_category,
            'confidence': assessment.confidence,
            'recommendation': assessment.recommendation,
            'findings': findings,
            'probabilities': {
                'scc': assessment.scc_probability,
                'bcc': assessment.bcc_probability,
                'melanoma': assessment.melanoma_probability,
                'benign': assessment.benign_probability
            }
        })
    
    @app.route('/api/generate-report', methods=['POST'])
    def generate_report():
        """Generate clinical report"""
        data = request.get_json() or {}
        patient_id = data.get('patient_id', 'TEST001')
        lesion_id = data.get('lesion_id', 'L001')
        
        # Quick analysis for report
        visual_data = visual_capture._simulated_capture_sequence(patient_id, lesion_id, "forearm")
        visual_features = visual_extractor.extract_all_features(visual_data)
        visual_vector = visual_extractor.generate_feature_vector(visual_features)
        
        thermal_data = thermal_system.capture_thermal_snapshot()
        thermal_features = thermal_system.extract_thermal_features(thermal_data)
        thermal_vector = thermal_system.generate_thermal_feature_vector(thermal_features)
        
        captures = ultrasound_hardware.capture_full_spectrum()
        acoustic_features = harmonic_analyzer.analyze_multi_frequency(captures)
        acoustic_vector = harmonic_analyzer.generate_feature_vector(acoustic_features)
        
        fused = fusion_engine.fuse_features(
            visual_features=visual_vector,
            thermal_features=thermal_vector,
            acoustic_features=acoustic_vector
        )
        
        assessment = fusion_engine.assess_risk(fused)
        
        report = fusion_engine.generate_comprehensive_report(
            assessment, fused,
            {'id': patient_id, 'location': lesion_id}
        )
        
        # Add acoustic analysis
        report += "\n\n" + harmonic_analyzer.explain_harmonic_findings(acoustic_features)
        
        return jsonify({
            'success': True,
            'report': report
        })
    
    return app


def run_app(host: str = '127.0.0.1', port: int = 5000, debug: bool = True):
    """Run the web application"""
    app = create_app()
    if app is not None:
        print(f"\n🔬 SCC Multi-Spectrum Detection System")
        print(f"   Starting server at http://{host}:{port}")
        print(f"   Press Ctrl+C to stop\n")
        app.run(host=host, port=port, debug=debug)
    else:
        print("Failed to create application")

