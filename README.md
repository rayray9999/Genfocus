# Generative Refocusing: Flexible Defocus Control from a Single Image

<div align="center">

[![Project Website](https://img.shields.io/badge/Project-Website-87CEEB?style=for-the-badge&logo=google-chrome&logoColor=white)](https://generative-refocusing.github.io/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/nycu-cplab/Genfocus-Demo)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-green?style=for-the-badge)](https://huggingface.co/nycu-cplab/Genfocus-Model)
[![License](https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge)](LICENSE)

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

### 2\. Download Weights

To run `demo.py`, you need to download the pre-trained models.

Please visit our [Hugging Face Model](https://huggingface.co/nycu-cplab/Genfocus-Model) and place the following files in the **root directory** of `Genfocus`:

| Filename / Folder | Description | Destination Path |
| :--- | :--- | :--- |
| `bokehNet.safetensors` | Main model for bokeh generation | `./Genfocus/bokehNet.safetensors` |
| `deblurNet.safetensors` | Model for image restoration | `./Genfocus/deblurNet.safetensors` |
| `checkpoint/` | Folder containing auxiliary checkpoints | `./Genfocus/checkpoint/` |

### 3\. Run Gradio Demo

Launch the interactive web interface locally:

```bash
python demo.py
```

The demo will be accessible at `http://127.0.0.1:7860` in your browser.

-----

## 🗺️ Roadmap & TODO

We are actively working on improving this project. Current progress:

  - [x] **Upload Model Weights**
  - [x] **Release HF Demo & Gradio Code** (with tiling tricks for high-res images)
  - [ ] **Release Benchmark data**
  - [ ] **Release Inference Code** (Support for adjustable parameters/settings)
  - [ ] **Release Training Code and Data**

-----

## 🔗 Citation

If you find this project useful for your research, please consider citing:

```bibtex
@article{Genfocus2025,
  title={Generative Refocusing: Flexible Defocus Control from a Single Image},
  author={Tuan Mu, Chun-Wei and Huang, Jia-Bin and Liu, Yu-Lun},
  journal={arXiv preprint},
  year={2025}
}
```

## 📧 Contact

For any questions or suggestions, please open an issue or contact me at [raytm9999.09@nycu.edu.tw](mailto:raytm9999.09@nycu.edu.tw).

<div align="center">
  <br>
  <p>Star 🌟 this repository if you like it!</p>
</div>