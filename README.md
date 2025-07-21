# 🧠 Stable Diffusion Multi-Prompt Image Generator

This Python script uses the Hugging Face 🤗 `diffusers` library to generate multiple images from user-defined text prompts using the `runwayml/stable-diffusion-v1-5` model. It supports GPU acceleration and visualizes the results in a single row using `matplotlib`.

---

## 📌 Description

The script loads the Stable Diffusion pipeline, feeds a list of prompts, and generates one image per prompt. Each image is then displayed with its corresponding caption (truncated for readability).

---

## 🖥️ Requirements

Make sure the following Python packages are installed:

```bash
pip install diffusers torch matplotlib
```

If you want GPU acceleration (recommended):
- Ensure you have an NVIDIA GPU
- Install `torch` with CUDA support (https://pytorch.org/get-started/locally/)

---

## 🧾 Example Script

```python
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load the Stable Diffusion pipeline with float16 precision for faster inference on GPU
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16  
)

# Move the pipeline to GPU if available, else CPU
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# List of prompts for image generation
prompts = [
    "a futuristic city with flying cars, ultra-detailed, 4K", 
    "a fantasy landscape with dragons and castles, vivid colors",
    "A formula 1 car racing on a track, high speed, dynamic angle",
]

# Generate and store images
images = []
for prompt in prompts:
    image = pipe(prompt).images[0]
    images.append(image)

# Plot the images with corresponding prompt captions
fig, axes = plt.subplots(1, len(prompts), figsize=(6 * len(prompts), 6))
for ax, img, prompt in zip(axes, images, prompts):
    ax.imshow(img)
    ax.set_title(prompt[:40] + '...' if len(prompt) > 40 else prompt, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

---

## 📷 Output

- Displays a horizontal gallery of AI-generated images.
- Each image corresponds to a user-provided prompt.
- Ideal for quick visualizations and prototyping image generation use cases.

---

## 🧠 Model Info

- **Model Name**: [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Library**: Hugging Face `diffusers`
- **Precision**: `float16` for GPU support (auto-detects CPU if needed)

---

## ⚠️ Notes

- First-time execution may take time due to model download (~4GB).
- GPU is highly recommended for performance.
- You can modify the `prompts` list to generate your own scenes.

---

## 📄 License

This code uses the Stable Diffusion model under the [CreativeML Open RAIL-M License](https://huggingface.co/spaces/CompVis/stable-diffusion-license). Please review the model's terms for commercial or public use.

---

## ✨ Author

Generated with ❤️ using Python, `diffusers`, and `matplotlib`.
