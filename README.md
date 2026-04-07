# 🔭 Fourier Optics & Computational Imaging
### *Machine Learning meets the Point Spread Function*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?logo=numpy)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

> **University Course** | Department of Physics  
> *Bridging classical wave optics, Fourier analysis, and modern machine learning for computational imaging systems.*

---

## 📖 About This Course

This course provides a rigorous treatment of **Fourier Optics** and its applications in **Computational Imaging**, with a central focus on the **Point Spread Function (PSF)** as the unifying mathematical object. Students will develop both analytical and computational intuition — from the diffraction integral to deep neural networks for image reconstruction.

The curriculum is structured around a core textbook on Fourier Optics and PSF theory, enriched with modern machine learning perspectives and hands-on computational labs.

---

## 🧭 Course Structure

```
📚 Foundations → 🌊 Wave Propagation → 🔬 PSF Theory → 🤖 ML Integration → 🛠️ Projects
```

### Part I — Mathematical Foundations

| Chapter | Topic | Key Concepts |
|--------|-------|-------------|
| 01 | Linear Systems & Signals | Convolution, impulse response, LTI systems |
| 02 | Fourier Analysis | DFT, FFT, Parseval's theorem, sampling |
| 03 | 2D Fourier Transform | Spatial frequencies, optical transfer function (OTF) |
| 04 | Complex Notation in Optics | Phasors, analytic signals, coherence |

### Part II — Fourier Optics

| Chapter | Topic | Key Concepts |
|--------|-------|-------------|
| 05 | Scalar Diffraction Theory | Huygens–Fresnel principle, Rayleigh–Sommerfeld |
| 06 | Fresnel & Fraunhofer Diffraction | Near-field, far-field, angular spectrum |
| 07 | Lens as a Fourier Transformer | Focal-plane transforms, 4f systems |
| 08 | Coherent & Incoherent Imaging | Coherent Transfer Function (CTF), OTF, MTF |

### Part III — Point Spread Function (PSF)

| Chapter | Topic | Key Concepts |
|--------|-------|-------------|
| 09 | PSF: Definition & Properties | Airy disk, diffraction limit, aberrations |
| 10 | PSF Measurement & Calibration | Bead experiments, blind deconvolution |
| 11 | Aberrations & Zernike Polynomials | Wavefront errors, pupil function |
| 12 | Spatially Varying PSF | Isoplanatism, anisoplanatic systems |

### Part IV — Machine Learning for Computational Imaging

| Chapter | Topic | Key Concepts |
|--------|-------|-------------|
| 13 | Classical Deconvolution | Wiener filter, Richardson–Lucy, regularization |
| 14 | Deep Learning Fundamentals | CNNs, U-Net, loss functions for imaging |
| 15 | PSF Estimation with Neural Networks | Blind & semi-blind PSF learning |
| 16 | End-to-End Optical System Design | Differentiable optics, physics-informed ML |
| 17 | Computational Microscopy | STORM, PALM, STED, deconvolution microscopy |
| 18 | Wavefront Sensing & Adaptive Optics | Shack-Hartmann, ML-based wavefront reconstruction |

### Part V — Advanced Projects

| Chapter | Topic | Key Concepts |
|--------|-------|-------------|
| 19 | Holography & Phase Retrieval | Gerchberg–Saxton, ptychography |
| 20 | Lensless Imaging | Diffuser cameras, coded apertures |
| 21 | Quantum Imaging (Introduction) | Ghost imaging, entangled photons, NOON states |

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.10+
- Basic knowledge of linear algebra and complex analysis
- Familiarity with NumPy/SciPy (recommended)

### Clone & Install

```bash
# Clone the repository
git clone https://github.com/<your-username>/fourier-optics-course.git
cd fourier-optics-course

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Launch Jupyter

```bash
jupyter lab
```

---

## 📦 Dependencies

```txt
numpy
scipy
matplotlib
scikit-image
torch
torchvision
jupyter
jupyterlab
tqdm
Pillow
zernpy          # Zernike polynomial utilities
aotools         # Adaptive optics tools
```

> Full list in [`requirements.txt`](./requirements.txt)

---

## 📂 Repository Structure

```
fourier-optics-course/
│
├── 📁 lectures/              # Slide decks and lecture notes (PDF)
│
├── 📁 notebooks/             # Jupyter notebooks per chapter
│   ├── ch01_linear_systems/
│   ├── ch05_diffraction/
│   ├── ch09_psf/
│   ├── ch14_deep_learning/
│   └── ...
│
├── 📁 datasets/              # Sample images, PSF calibration data
│
├── 📁 src/                   # Reusable Python modules
│   ├── fourier_utils.py      # FFT wrappers, frequency grids
│   ├── psf_tools.py          # PSF generation, fitting, metrics
│   ├── deconvolution.py      # Classical and ML-based deconvolution
│   └── optics_sim.py         # Wave propagation simulators
│
├── 📁 projects/              # Student project templates
│   ├── project_01_psf_estimation/
│   ├── project_02_deconvolution_net/
│   └── project_03_lensless_imaging/
│
├── 📁 solutions/             # Reference solutions (instructor access)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧪 Core Simulations Included

- **Fresnel & Fraunhofer diffraction** propagator
- **PSF generator** from Zernike-expanded pupil functions
- **4f optical system** simulator with custom aberrations
- **Wiener deconvolution** with noise estimation
- **U-Net** for blind PSF estimation (PyTorch)
- **Phase retrieval** via Gerchberg–Saxton algorithm
- **Wavefront reconstruction** using neural networks

---

## 📐 Key Equations at a Glance

$$U(x, y) = \mathcal{F}^{-1}\left\{ \tilde{U}(f_x, f_y) \cdot H(f_x, f_y) \right\}$$

$$\text{PSF}(\mathbf{r}) = \left| \mathcal{F}\left\{ P(\mathbf{u}) \, e^{i W(\mathbf{u})} \right\} \right|^2$$

$$\text{OTF}(f_x, f_y) = \mathcal{F}\left\{ \text{PSF}(x, y) \right\}$$

$$\hat{F}(f) = \frac{G(f) \cdot H^*(f)}{|H(f)|^2 + S_n(f)/S_f(f)} \quad \text{(Wiener filter)}$$

---

## 🎓 Learning Outcomes

By the end of this course, students will be able to:

- [ ] Derive and compute diffraction patterns analytically and numerically
- [ ] Characterize optical systems through PSF measurement and modeling
- [ ] Decompose wavefront aberrations using Zernike polynomials
- [ ] Implement classical deconvolution algorithms (Wiener, RL)
- [ ] Design and train neural networks for image restoration
- [ ] Apply end-to-end differentiable optics to system optimization
- [ ] Analyze modern computational imaging architectures

---

## 📊 Evaluation

| Component | Weight |
|-----------|--------|
| Problem Sets (6 total) | 30% |
| Computational Labs | 20% |
| Midterm Exam | 20% |
| Final Project | 30% |

---

## 📚 References

1. **Goodman, J. W.** — *Introduction to Fourier Optics* (4th ed.), W. H. Freeman, 2017.
2. **Born, M. & Wolf, E.** — *Principles of Optics*, Cambridge University Press.
3. **Saleh, B. & Teich, M.** — *Fundamentals of Photonics*, Wiley.
4. **LeCun, Y. et al.** — *Deep Learning*, MIT Press, 2016.
5. **Sitzmann, V. et al.** — *End-to-end Optimization of Optics and Image Processing*, SIGGRAPH 2018.
6. **Ongie, G. et al.** — *Deep Learning Techniques for Inverse Problems in Imaging*, IEEE JSTSP, 2020.

---

## 🤝 Contributing

Contributions, corrections, and improvements are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-topic`
3. Commit your changes: `git commit -m "Add: description"`
4. Push and open a Pull Request

Please follow the [Code of Conduct](./CODE_OF_CONDUCT.md) and keep notebooks clean before submitting.

---

## 📄 License

This course material is released under the [MIT License](./LICENSE).  
© 2025 – Universidad · Department of Physics

---

<div align="center">

*"The Fourier transform does not merely describe light — it is the language in which optical systems think."*

**⭐ Star this repo if you find it useful!**

</div>
