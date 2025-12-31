# Generative Refocusing: Flexible Defocus Control from a Single Image

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green?logo=googlechrome&logoColor=green)](https://generative-refocusing.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2512.16923)
[![YouTube Tutorial](https://img.shields.io/badge/YouTube-Tutorial-red?logo=youtube&logoColor=red)](https://youtu.be/CMh_jGDl-RE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/nycu-cplab/Genfocus-Demo)


<div align="center">
  <img src="./assets/demo_vid.gif" width="50%" alt="Demo Video">
</div>

</div>

---

## ⚡ Quick Start

Follow the steps below to set up the environment and run the inference demo.

### 1. Installation

Clone the repository:

```bash
git clone git@github.com:rayray9999/Genfocus.git
cd Genfocus
````

Environment setup:

```bash
conda create -n Genfocus python=3.12
conda activate Genfocus
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 2. Download Weights

You can download the pre-trained models using the following commands. Ensure you are in the `Genfocus` root directory.

```bash
# 1. Download main models to the root directory
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/bokehNet.safetensors
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/deblurNet.safetensors

# 2. Setup checkpoints directory and download auxiliary model
mkdir -p checkpoints
cd checkpoints
wget https://huggingface.co/nycu-cplab/Genfocus-Model/resolve/main/checkpoints/depth_pro.pt
cd ..
```
### 3\. Run Gradio Demo

Launch the interactive web interface locally:  
> **Note:** This project uses [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev). You must request access and authenticate locally before running the demo.  
> ⚠️ **VRAM warning:** GPU memory usage can be high. On an NVIDIA A6000, peak usage may reach ~45GB depending on resolution and settings.  
> **Tip (Advanced Settings):** If you want faster inference, try lowering **num_inference_steps**, resizing the input in **Advanced Settings**.

```bash
python demo.py


The demo will be accessible at `http://127.0.0.1:7860` in your browser.

-----

## 🗺️ Roadmap & TODO

We are actively working on improving this project. Current progress:

  - [x] **Upload Model Weights**
  - [x] **Release HF Demo & Gradio Code** (with tiling tricks for high-res images)
  - [ ] **Release Inference Code** (Support for adjustable parameters/settings)
  - [ ] **Release ComfyUI Workflow / Node**
  - [ ] **Release Benchmark data**
  - [ ] **Release Training Code and Data**


-----

## 🔗 Citation

If you find this project useful for your research, please consider citing:

```bibtex
@article{Genfocus2025,
  title={Generative Refocusing: Flexible Defocus Control from a Single Image},
  author={Tuan Mu, Chun-Wei and Huang, Jia-Bin and Liu, Yu-Lun},
  journal={arXiv preprint arXiv:2512.16923},
  year={2025}
}
```

## 📧 Contact

For any questions or suggestions, please open an issue or contact me at [raytm9999.cs09@nycu.edu.tw](mailto:raytm9999.cs09@nycu.edu.tw).

<div align="center">
  <br>
  <p>Star 🌟 this repository if you like it!</p>
</div>
