---
description: 'Core architectural rules, mathematical constraints, and coding guidelines for the 2D Kinematic Gait Analysis System.'
applyTo: '*.py, src/**/*.py, ui/**/*.py, main.py'
---

# Project Context
This project is an undergraduate Computer Science thesis focused on developing a 2D Kinematic Gait Analysis System. It processes sagittal-view video feeds (both live webcam and pre-recorded MP4s) using markerless pose estimation to quantify bilateral knee flexion asymmetry. The code must meet rigorous academic and clinical software standards.

# Tech Stack & Environment
- **Language:** Python 3.12 (Strict type hinting is required for all functions).
- **Environment:** Native local execution on Windows 11. All file paths must remain OS-agnostic (use `os.path` or `pathlib` to prevent Windows backslash escape sequence errors). GUI elements (like `cv2.imshow` or Tkinter) will render natively via Windows. For video capture, prioritize low-latency Windows backends (e.g., explicitly testing `cv2.CAP_DSHOW` or `cv2.CAP_MSMF` if the default index `0` lags).
- **Core Libraries:** opencv-python (4.9.0.80), mediapipe (0.10.14), numpy (1.26.4), scipy (1.12.0), pandas (2.2.1).

# Architectural Pipeline (Do Not Violate)
The system is divided into four strictly decoupled modules. Code must belong to its specific region:
1. **Data Ingestion (`src/ingestion.py`):** Handles OpenCV video capture. Must yield discrete frames at a target of 30 FPS.
2. **Pose Estimation (`src/pose_estimation.py`):** Implements MediaPipe BlazePose. Must extract spatial coordinates (x, y) and visibility scores (c) for the hip, knee, and ankle.
3. **Filter & Computation (`src/kinematics.py`):** Handles noise reduction (SciPy) and angular vector trigonometry (math).
4. **Classification (`src/classification.py`):** Calculates Robinson's Symmetry Index and triggers the final binary diagnosis.

# Strict Mathematical & Logic Constraints
When generating algorithms, you must strictly adhere to these specific formulas. Do not substitute them with generic machine learning functions.

1. **Visibility Thresholding:** - Any MediaPipe landmark with a confidence score of `c < 0.5` must be aggressively dropped/ignored before filtering.
2. **Signal Smoothing:** - Apply SciPy's Savitzky-Golay (`scipy.signal.savgol_filter`) low-pass filter to the coordinate arrays.
3. **Kinematic Angle Computation:** - Use the 2-argument inverse tangent function (`math.atan2` or `numpy.arctan2`) to calculate the interior knee flexion angle.
   - Core Logic: `theta = abs((atan2(y_ankle - y_knee, x_ankle - x_knee) - atan2(y_hip - y_knee, x_hip - x_knee)) * 180 / pi)`
   - Normalization Rule: If `theta > 180`, it must be adjusted by `360 - theta` to reflect human biological limits.
4. **Symmetry Quantification:** - Calculate Robinson’s Symmetry Index (SI).
   - Crucial Defense: You MUST apply absolute magnitudes to both X_R and X_L in the denominator to prevent zero-crossing division errors.
   - Core Logic: `SI = (abs(X_R - X_L) / (0.5 * (abs(X_R) + abs(X_L)))) * 100`
5. **Clinical Threshold:** - `SI >= 10%` evaluates to True/Abnormal (1).
   - `SI < 10%` evaluates to False/Normal (0).

# Coding Guidelines
- **Performance First:** The system must process video continuously without heavy latency. Avoid nested loops where vectorized NumPy operations can be used.
- **Documentation:** Every function must have a complete docstring (Google or Sphinx format) explaining its inputs, outputs, and the academic justification for the math used inside it.
- **No Monoliths:** Do not put processing logic inside UI files. The `ui/dashboard.py` should only handle display, calling imports from the `src/` directory.