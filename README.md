<div align="center">

# Magic Canvas 📷 🖼️

</div>
Website to create an image on CPU (GPU recommended) 🌐

This repository provides a simple website / Jupyter-based project to generate images using multiple Stable Diffusion style models. The project was developed to work on CPU (for accessibility/testing) but will be significantly faster on a GPU — GPU usage is recommended for practical generation times.

What I did and what's next 💪

I created the core model list and integrated local persistence so every generated image, its prompt, and the generation time are stored in a single SQLite database file (`ai_images`). I also tracked model download entries in the same DB. My next steps are to improve performance (GPU support/optimization), add more automated tests and examples, and refine the web UI for better UX.

Features ✨
- Six image-generation model configurations included (various size/speed/quality profiles)
- Every generated image (binary/filename), prompt text, generation timestamp, and model used is stored in an `ai_images` SQLite database file
- Tracks model downloads in the DB
- Designed to run on CPU but recommends GPU for speed and quality

Repository 🗃️
- Owner / repo: Shashankkusu/AI_Image_Gen
- Technology stack: Python (Flask) , Javascript,html,css
- DB file: ai_images (SQLite)

Models 🤖
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

Quick start 🚀 (local, CPU)
1. Clone the repo 📥
   - git clone https://github.com/Shashankkusu/AI_Image_Gen.git
2. Switch to the project directory 📁
   - cd AI_Image_Gen
3. Install dependencies 📦
   - 💡 Recommended: create a virtualenv first
   - pip install -r requirements.txt
   - 🛠️ (If there is no requirements.txt, install commonly used libs: `pip install torch torchvision transformers diffusers accelerate pillow jupyterlab sqlite3` — ℹ️ note `sqlite3` may be bundled with Python.)
4. Run from Jupyter Notebook 📓
   - jupyter lab  (or jupyter notebook) and open the provided notebooks (for research) 📂
5. Or run app script 🖥️
   - python app.py
   - The web interface 🌐 will start and allow prompt-based generation.

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

Database: ai_images (SQLite) 🗄️

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

## Example Usage Flow 🔄

1. **User selects a model** (e.g., "small") and a resolution and enters a prompt ✍️
2. **The backend enqueues or directly runs** the generation with the corresponding model parameters ⚙️
3. **When generation completes**, the resulting image is saved to disk (e.g. `./outputs/`) and metadata is inserted into `generated_images` 💾
4. **The UI shows the generated image** and provides options to download or regenerate 📱

## Performance Notes ⚡

- **CPU**: Works for experimentation but generation will be slow, especially for larger models (several minutes per image depending on CPU and resolution) 🐌
- **GPU**: Strongly recommended! If you have an NVIDIA GPU and proper CUDA drivers, install the appropriate `torch` build with CUDA and enable GPU generation 🚀
- **Memory/Disk**: Some models are multiple GBs — ensure enough disk space and memory for model loading 💽

## ⚠️ A Quick, Very Honest Performance Note 

My machine has the computational power of a confused toaster trying to run a space shuttle 🍞🚀 — so images take a while, and quality/consistency will vary. If you expect blazing results on CPU, well… don't. I tried. The toaster lost. 😅

<div align="center">
  <img width="526" height="583" alt="Performance meme" src="https://github.com/user-attachments/assets/321f78c9-7cb2-4750-83fe-b42dccdadcfc" />
</div>





## Research Areas Related to This Project 🔬

- **GAN** (Generative Adversarial Networks) 🎭
- **ViT** (Vision Transformers) 👁️
- **Transformers** (as general sequence models and attention mechanisms) 🧠
- **Stable Diffusion** (denoising diffusion probabilistic models used for image generation) 🌪️

## Troubleshooting Tips 🛠️

- **"Out of memory" errors**: Reduce resolution, switch to a smaller model (tiny/mini-sd), or use a machine with more RAM/GPU memory 🚫💾
- **Slow CPU generation**: Reduce steps or batch size, or run on GPU 🐢
- **Missing dependencies**: Double-check the Python environment and installed packages 📦

## Contributing 🤝

- **Contributions welcome!** Please open issues or pull requests 🙂
- **If you add models**, please include source, size, and expected resolution in the MODELS dict 📝

## Contact 📧

- **Repo owner**: Shashankkusu [gowrisesharoa@gmail.com] 👨‍💻

