# Image Q&A: BLIP or CLIP with Local LLM (Ollama)

This project is a **Streamlit web application** for image-based question answering. It uses:
- **BLIP** for image captioning
- **CLIP** for image-text similarity matching
- A **local LLM (Llama2 via Ollama)** for answering user questions about images

You can upload an image, generate captions, compare with descriptions, and ask questions—all powered by open-source models running locally.

## Features

- **Image Upload:** Supports PNG, JPG, and JPEG files.
- **BLIP Captioning:** Generates a descriptive caption for your image.
- **CLIP Matching:** Compares your image to user-provided descriptions and finds the best match.
- **Local LLM Q&A:** Ask questions about the image; the LLM answers using BLIP/CLIP context.
- **Runs Locally:** All models run on your machine—no cloud required.

## Demo

![Screenshot 01](https://github.com/user-attachments/assets/b9864581-1b10-425d-818c-ec1321d6ce39)

![Screenshot 05](https://github.com/user-attachments/assets/258bb368-91a8-4326-bde6-17761745bb08)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Ollama (for running the Llama2 model locally)
- Windows (instructions here; Linux/Mac can be adapted)

### Install Ollama and Llama2

1. Download and install Ollama for Windows from [here](https://ollama.com/download/windows)
2. Open a terminal and run:
   
```bash
ollama pull llama2
ollama serve
```

### Set Up Python Environment

```bash
python -m venv my-env
my-env\Scripts\activate # for Windows
pip install --upgrade pip
pip install streamlit transformers torch pillow ollama
```

### Run the App

```bash
streamlit run app.py  # only for local PC
streamlit run app.py --server.address=0.0.0.0  # if you want to run it in your local PC as well as in your smartphone
```

- The `--server.address=0.0.0.0` flag allows access from other devices on your local network.

## Accessing from Your Smartphone

1. Make sure your PC and phone are connected to the same WiFi network.
2. Find your PC’s local IP address (e.g., `192.168.0.187`).
3. On your phone’s browser, navigate to:

```bash
http://<your-pc-ip>:8501  # Ex: https://192.168.0.187:8501
```
4. If connection fails, check Windows Firewall settings to allow inbound connections on port 8501.

## Usage

1. Upload an image (PNG, JPG, JPEG).
2. Choose a mode:
- **BLIP:** Generate a caption describing the image.
- **CLIP:** Enter descriptions (one per line) and find the best match.
3. Ask a question about the image and get answers from the local LLM.
4. View captions, best matches, and LLM answers in the app.

## File Structure

| File           | Description                          |
|----------------|------------------------------------|
| `data`    | Sample Images   |
| `output`       | Output from `app.py` file          |
| `README.md`    | This file                         |
| `app.py`       | Main Streamlit application          |
| `requirements.txt` | Python dependencies               |

## Requirements

```bash
streamlit
transformers
torch
pillow
ollama
```

## Troubleshooting

- **Ollama not running:** Ensure `ollama serve` is active and the Llama2 model is pulled.
- **Firewall issues:** Allow port 8501 in Windows Firewall inbound rules.
- **Network issues:** Confirm both devices are on the same network and no VPNs interfere.
- **Model errors:** Verify Python environment and package installation.

## Credits

- BLIP: [Salesforce/blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base)
- CLIP: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- LLM: Llama2 via [Ollama](https://ollama.com/download)

## Acknowledgments

- Streamlit for fast app development.
- HuggingFace Transformers for model access.
- Ollama for easy local LLM deployment.

---

Feel free to open issues or submit pull requests to improve this project!

