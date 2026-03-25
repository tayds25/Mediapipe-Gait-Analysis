# Markerless 2D Kinematic Gait Analysis System using Mediapipe

### Overview
This repository provides the source code for a computer vision-based kinematic gait analysis system. Designed for a Computer Science thesis study, the system employs a markerless two-dimensional pose estimation pipeline to quantify bilateral knee flexion asymmetry. By analyzing sagittal-view video feeds from both live webcam captures and pre-recorded datasets, the algorithm computes Robinson’s Symmetry Index (SI) to classify walking patterns as either normal or abnormal.

### Conceptual Architecture
The system uses a continuous, algorithmic data pipeline that processes raw pixel data and outputs a binary classification.

1. **Data Ingestion:** Extracts discrete frames from raw video sequences using OpenCV.
2. **Pose Estimation:** Utilizes MediaPipe to extract spatial coordinates $(x, y)$ and visibility confidence scores $(c)$ for the hip, knee, and ankle joints.
3. **Filtering & Computation:** Applies a visibility threshold (dropping data where $c_i < 0.5$) and a SciPy low-pass filter to smooth coordinate jitter. The interior knee flexion angle is computed using vector trigonometry:
   $$\theta_{flexion} = \left| \left( \arctan2(y_a - y_k, x_a - x_k) - \arctan2(y_h - y_k, x_h - x_k) \right) \times \frac{180}{\pi} \right|$$
4. **Classification:** Parses the gait cycle to find peak flexion magnitudes for the right ($X_R$) and left ($X_L$) limbs. Calculates Robinson's Symmetry Index (SI):
   $$SI = \frac{|X_R - X_L|}{0.5(|X_R| + |X_L|)} \times 100\%$$
   The system triggers an "Abnormal Gait" alert if the resulting $SI \ge 10\%$.

### Prerequisites
The pipeline is optimized for local development and hardware execution to achieve zero-latency video processing and precise real-time frames-per-second (FPS) measurements.

* **Operating System:** Windows 11 or Linux.
* **Hardware:** Native webcam access for primary live trials. Minimum 480p resolution.
* **Language:** Python 3.12
* **Core Libraries:** `opencv-python`, `mediapipe`, `numpy`, `scipy`, `pandas`

### Repository Structure
```text
Kinematic-Gait-Analysis/
├── data/                      # (Git-ignored) Datasets
│   ├── raw_gavd_files/          # Video files from GAVD
│   ├── live_recordings/         # Mirrored MP4s from live trials
├── src/                       # Core computational pipeline
│   ├── ingestion.py             # OpenCV frame extraction logic
│   ├── pose_estimation.py       # MediaPipe BlazePose integration
│   ├── kinematics.py            # SciPy filters & angular math
│   └── classification.py        # Symmetry Index and threshold logic
├── ui/                        # Graphical User Interface assets
│   └── dashboard.py             # Application window
├── docs/                      # Documentation & validation rubrics
├── .gitignore                 # Untracked files configuration
├── requirements.txt           # Python dependency tree
└── main.py                    # Central execution script
```

---

### Setup & Development

**Step 1: Clone the Repository**  
Open your terminal and clone the project to your local machine:
```
git clone https://github.com/tayds25/Mediapipe-Gait-Analysis.git
cd Kinematic-Gait-Analysis
```

**Step 2: Open in VS Code**  
Launch Visual Studio Code in the current directory:
```
code .
```
In VS Code, open the integrated terminal `(Ctrl + ~)`.

**Step 3: Create the Virtual Environment**  
Create an isolated Python 3.12 environment:
```
python -m venv venv
```

**Step 4: Activate the Environment**  
```
# For Windows
.\.venv\Scripts\activate
```

**Step 5: Install Dependencies**  
```
pip install -r requirements.txt
```

**Step 6: Switch to the Development Branch**  
To avoid merge conflicts, ensure you are working on the active `dev` branch:
```
git fetch
git checkout dev
```

---

### Authors
* Cruz, Raymond Lorenzo B.
* Delos Santos, Tayshaun M.
* Ocampo, Gio Fernando A.
