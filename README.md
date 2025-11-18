<div align="center">

# Magic Canvas 📷 🖼️

</div>
Website and Jupyter Notebook examples to generate images locally (CPU-first, GPU recommended)  
Repository description: Website to create an image on CPU but suggest to use GPU. ⚠️ GPU recommended for reasonable speed.

This README explains the project, architecture, how to set it up, model download instructions, hardware guidance, usage examples and prompts, technology stack, prompt engineering tips, known limitations, and ideas for future improvements. ✨

---

## Table of contents 📚
- Project overview
- Architecture 🏗️
- Hardware requirements 💻🧮
- Setup & installation ⚙️
  - Python environment 🐍
  - Installing dependencies 📥
  - Downloading models 💾
  - Environment variables and tokens 🔒
- Running the website and notebooks ▶️
- Usage examples & example prompts ✍️
- Technology stack & model details 🧰
- Prompt engineering tips & best practices 🧠
- Limitations ⚠️
- Future improvements 🛠️
- License & credits 📝

---

## Project overview ✨

AI_Image_Gen is a small project that demonstrates how to run a generative image model locally and expose a simple website (and Jupyter notebooks) to create images. The repo focuses on CPU-first usability so it can run on machines without a GPU, but GPU acceleration is strongly recommended for practical speed and larger images. 🚀

Use cases:
- Experiment with local image generation without external APIs 🛡️
- Prototype prompts and visual styles 🎨
- Educational demos using Jupyter notebooks 🧪

---

## Architecture 🏗️

High-level components:
- Jupyter Notebooks: interactive examples and experiments (primary code in notebooks) 📓
- Local model loader: downloads/loads a diffusion model into memory (Hugging Face / diffusers example) 📥
- Inference pipeline: accepts text prompts + parameters and returns generated images (PIL/PNG) 🖼️
- Simple web UI: minimal website (Flask / Streamlit) to accept prompts and display results 🌐

Data / flow:
1. User enters a prompt in notebook or web UI ✍️  
2. Server/inference code tokenizes prompt and runs the diffusion pipeline 🔁  
3. Generated image(s) returned to the client and saved to disk (or displayed inline) 💾

---

## Hardware requirements 💻

CPU-only (possible but slow):
- 4+ CPU cores recommended 🧮
- 16GB+ RAM recommended (some models require more) 🧠
- Swap space helpful when RAM is limited 🗃️

GPU (recommended) 🔥:
- NVIDIA GPU with CUDA preferred (for PyTorch/CUDA builds) 🧪
- 8GB VRAM minimum for small images (512×512) ⚖️
- 12+ GB VRAM recommended for larger images or higher batch sizes 🚀
- Use recent drivers + CUDA matching your PyTorch build 🧰

Notes:
- CPU generation for 512×512 can take tens of seconds to minutes depending on hardware ⏱️
- Use smaller/distilled models for low-memory environments 🧩

---
## ⚠️ A Quick, Very Honest Performance Note 

My machine has the computational power of a confused toaster trying to run a space shuttle 🍞🚀 — so images take a while, and quality/consistency will vary. If you expect blazing results on CPU, well… don't. I tried. The toaster lost. 😅

<div align="center">
  <img width="526" height="583" alt="Performance meme" src="https://github.com/user-attachments/assets/321f78c9-7cb2-4750-83fe-b42dccdadcfc" />
</div>
Common models:
- Stable Diffusion v1.4 / v1.5 (latent diffusion)
- Stable Diffusion XL (higher quality)
- Other diffusion or GAN-based models supported with adapters

Model notes:
- Some models require acceptance of licenses or authentication tokens 🔒
- safetensors is recommended for safer/smaller files 🧾

---

## Prompt engineering tips & best practices 🧠

- Be specific: include style, camera terms (lens/aperture), lighting, mood, and color palette 🎨
- Use adjectives: "ultra-detailed", "photorealistic", "cinematic" ✨
- Use negative prompts to filter undesired traits: "lowres, watermark, blurry" ❌
- Start simple and iterate — add details gradually ♻️
- Use seeds for reproducible results: generator=torch.Generator(device).manual_seed(SEED) 🔁
- For portraits, include camera/lens details: "50mm, f/1.8, rim light" 📷
- Respect model and copyright policies when invoking artists' names ⚖️

Advanced:
- Prompt chaining: generate variations, pick the best, refine prompts 🔬
- Combine conditioning: text + image (img2img) for guided edits 🖼️➡️🖼️
- Blend prompts or use weighted prompts where supported ⚖️

---

## Limitations ⚠️

- CPU performance is much slower than GPU — expect long runtimes on CPU ⏱️
- Memory: large models may not fit in RAM or GPU VRAM; reduce resolution/batch size 🧩
- Safety: models can produce problematic content — add moderation in production 🛡️
- Artifacts & hallucinations: common issues (odd hands, misplaced text) 🌀
- Licensing: check model pages for usage restrictions 🔍
- Results can vary across hardware and numeric backends 🌐

---

## Future improvements 🔮

Ideas to extend the project:
- Provide Docker images with CUDA support for easy GPU deployment 🐳
- Add fine-tuning tools (LoRA / DreamBooth) to adapt models to custom datasets 🧰
- Style-transfer and model blending features (mix CLIP embeddings) 🎭
- Web UI enhancements: progress bars, galleries, user accounts, seed saving 🖥️
- Batch generation, queues, and multi-user scheduling ⚙️
- Safety moderation pipeline and content filters 🛡️
- Memory optimizations: xformers, sliced attention, torch.compile (where available) ⚡
- Add img2img, inpainting, and mask-based editing features ✂️

---

## Quick checklist to get started ✅

1. Clone repo 📥  
2. Create virtualenv and install dependencies 🐍  
3. Download a supported model and set HUGGINGFACE_TOKEN if required 🔑  
4. Run Jupyter and open the sample notebook to test 📓  
5. Optionally run the web app (Streamlit/Flask) 🌐

---

## Models 🤖

The repository exposes the following model configurations (MODELS dictionary):

- 🐭 tiny
  - name: OFA-Sys/small-stable-diffusion-v0
  - size: 400MB 💾
  - speed: Fast ⚡
  - quality: Good 👍
  - resolution: 384
  - type: stable_diffusion

- 🔵 small
  - name: runwayml/stable-diffusion-v1-5
  - size: 7GB 💾
  - speed: Medium 🐢
  - quality: Excellent 🌟
  - resolution: 512
  - type: stable_diffusion

- ✨ dreamshaper
  - name: Lykon/DreamShaper
  - size: 5GB 💾
  - speed: Medium 🐢
  - quality: Excellent 🌟
  - resolution: 512
  - type: stable_diffusion

- 🎨 openjourney
  - name: prompthero/openjourney-v4
  - size: 4GB 💾
  - speed: Medium 🐢
  - quality: Very Good 💫
  - resolution: 512
  - type: stable_diffusion

- mini-sd
  - name: OFA-Sys/small-stable-diffusion-v0
  - size: 400MB 💾
  - speed: Very Fast 🚀
  - quality: Good 👍
  - resolution: 256
  - type: stable_diffusion

- portrait
  - name: wavymulder/portraitplus
  - size: 2GB 💾
  - speed: Fast ⚡
  - quality: Very Good 💫
  - resolution: 384
  - type: stable_diffusion
---
## Quick start 🚀 (Local — CPU)

1. Clone the repo 📥
```bash
git clone https://github.com/Shashankkusu/AI_Image_Gen.git
cd AI_Image_Gen
```

2. Create and activate a Python virtual environment (recommended) 🐍
```bash
# Create venv
python3 -m venv .venv

# Activate on macOS / Linux
source .venv/bin/activate

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Or on Windows (Command Prompt)
.venv\Scripts\activate.bat
```

3. Install dependencies 📦
- If the repository includes a requirements file:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
- If there is no `requirements.txt`, install commonly used libraries:
```bash
pip install --upgrade pip
pip install torch torchvision transformers diffusers accelerate pillow jupyterlab
```
ℹ️ Note: `sqlite3` is typically included with the standard Python distribution; if it's missing, install it via your OS package manager (e.g., `apt`, `brew`) rather than pip.

4. Run from Jupyter Notebook 📓
```bash
jupyter lab    # or: jupyter notebook
```
Then open the provided notebooks in the browser.

5. Or run the app script 🖥️
```bash
python app.py
```
The web interface will start and print a local URL (e.g., http://127.0.0.1:7860 or another port). Open that URL in your browser to use the prompt-based generator.

Notes & tips:
- CPU-only runs are possible but significantly slower than GPU — use a GPU for faster generation when available. ⚠️
- If the project uses a different web entrypoint (e.g., `streamlit` or `flask`), check the repo files for exact commands (e.g., `streamlit run app_streamlit.py` or `FLASK_APP=app.py flask run`). 🔎
- If models require authentication (Hugging Face tokens), set the required environment variables before running the app.
---
Using the MODELS dictionary 🗂️
- The MODELS mapping in the code contains keys (tiny, small, dreamshaper, openjourney, mini-sd, portrait).
- ➕ To add a model: update the MODELS dict with a new key and required attributes (`name`, `size`, `speed`, `quality`, `resolution`, `type`).
- Example:
  MODELS["my-model"] = {
    "name": "organization/model-name",
    "size": "3GB",
    "speed": "Medium",
    "quality": "Very Good",
    "resolution": 512,
    "type": "stable_diffusion"
  }
---
## Database: ai_images (SQLite) 🗄️

All generated images 🖼️, prompts ✍️, generation times ⏰, and model download metadata are stored in an SQLite file named ai_images. The DB contains three tables:

1) generated_images 📊
-Stores each generated image metadata (filename or blob), prompt, model used, and timestamp.

📋 Table Structure:
<img width="1457" height="138" alt="image" src="https://github.com/user-attachments/assets/fd26bea0-8b95-4e05-8c45-9bc78688cb14" />


2) model_downloads
- Tracks downloads (or loads) of model files, useful for debugging and telemetry.

📋 Table Structure:
<img width="493" height="139" alt="image" src="https://github.com/user-attachments/assets/8f1a87ce-7226-4ae5-aa09-882579f5edf5" />


3) sqllite_sequence
- Standard SQLite sequence table used for AUTOINCREMENT bookkeeping (created automatically by SQLite).

📋 Table Structure:
<img width="211" height="113" alt="image" src="https://github.com/user-attachments/assets/3f72a077-1b8d-413f-98c8-8ccff9cc6162" />
---
## Example Usage Flow 🔄

1. **User selects a model** (e.g., "small") and a resolution and enters a prompt ✍️
2. **The backend enqueues or directly runs** the generation with the corresponding model parameters ⚙️
3. **When generation completes**, the resulting image is saved to disk (e.g. `./outputs/`) and metadata is inserted into `generated_images` 💾
4. **The UI shows the generated image** and provides options to download or regenerate 📱

---
## Research Areas Related to This Project 🔬

- **GAN** (Generative Adversarial Networks) 🎭
- **ViT** (Vision Transformers) 👁️
- **Transformers** (as general sequence models and attention mechanisms) 🧠
- **Stable Diffusion** (denoising diffusion probabilistic models used for image generation) 🌪️
---
## Troubleshooting Tips 🛠️

- **"Out of memory" errors**: Reduce resolution, switch to a smaller model (tiny/mini-sd), or use a machine with more RAM/GPU memory 🚫💾
- **Slow CPU generation**: Reduce steps or batch size, or run on GPU 🐢
- **Missing dependencies**: Double-check the Python environment and installed packages 📦
---

## License & credits 📝

- Check the `LICENSE` file in this repository for project license 📜  
- Models and pre-trained weights often have separate licenses — see the model provider page 🔗  
- Credits: Hugging Face diffusers, PyTorch community, model publishers 🙏
- 
---
## Contact 📧

- **Repo owner**: Shashankkusu [gowrisesharoa@gmail.com] 👨‍💻

