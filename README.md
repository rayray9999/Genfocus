抱歉，讓你感到困擾了！這是一個誤會，我剛剛雖然有放進程式碼區塊，但可能因為排版關係讓你覺得像是普通文字。

這裡我直接提供**純粹的原始碼 (Raw Markdown Source)**，請直接點擊下方黑色區塊右上角的 **"Copy"** 按鈕，這就是完整的 `README.md` 檔案內容：

````markdown
# Generative Refocusing: Flexible Defocus Control from a Single Image

<div align="center">

[![Project Website](https://img.shields.io/badge/Project-Website-87CEEB?style=for-the-badge&logo=google-chrome&logoColor=white)](https://generative-refocusing.github.io/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/nycu-cplab/Genfocus-Demo)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-green?style=for-the-badge)](https://huggingface.co/nycu-cplab/Genfocus-Model)
[![License](https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge)](LICENSE)

<h1>Generative Refocusing</h1>
<h3>Flexible Defocus Control from a Single Image</h3>

<p align="center">
  A novel framework for synthesizing realistic bokeh and refocusing effects from a single image using generative priors.
</p>

<div style="width: 100%; text-align: center; margin: 20px 0;">
    <video width="100%" autoplay loop muted playsinline>
        <source src="./assets/demo_vid.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>

</div>

---

## ⚡ Quick Start

Follow the steps below to set up the environment and run the inference demo.

### 1. Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone git@github.com:rayray9999/Genfocus.git
cd Genfocus

# Install requirements
pip install -r requirements.txt
````

### 2\. Download Weights

To run `demo.py`, you need to download the pre-trained models.
Please visit our [Hugging Face Model](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/nycu-cplab/Genfocus-Model) and place the following files in the **root directory** of `Genfocus`:

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

For any questions or suggestions, please open an issue or contact me at raytm9999.09@nycu.edu.tw.

\<div align="center"\>
<br>
\<p\>Star 🌟 this repository if you like it\!\</p\>
\</div\>

```
```