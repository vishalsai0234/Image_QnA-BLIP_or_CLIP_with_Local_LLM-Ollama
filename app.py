import streamlit as st
from PIL import Image
import ollama
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

# === Config ===
MODEL_NAME = "llama2"  # Ollama local LLM model name

# === Load Models with caching ===
@st.cache_resource(show_spinner=False)
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource(show_spinner=False)
def load_clip_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model

blip_processor, blip_model = load_blip_model()
clip_processor, clip_model = load_clip_model()

# === Streamlit UI ===
st.title("Image Q&A: BLIP or CLIP with Local LLM (Ollama)")

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Choose mode
    mode = st.radio("Choose mode:", options=["BLIP (Caption Generation)", "CLIP (Description Matching)"])

    # Initialize session state
    if "blip_caption" not in st.session_state:
        st.session_state.blip_caption = ""
    if "clip_scores" not in st.session_state:
        st.session_state.clip_scores = []
    if "clip_best" not in st.session_state:
        st.session_state.clip_best = ""

    if mode == "BLIP (Caption Generation)":
        if st.button("Generate Caption"):
            inputs = blip_processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            st.session_state.blip_caption = caption
            st.success(f"BLIP Caption: {caption}")

    else:  # CLIP mode
        # User inputs descriptions for CLIP matching
        user_descriptions = st.text_area(
            "Enter descriptions/sentences for CLIP to compare (one per line):",
            value="a photo of a cat\na photo of a dog\na photo of a mountain\na photo of a beach"
        )
        descriptions = [line.strip() for line in user_descriptions.split("\n") if line.strip()]

        if st.button("Run CLIP Description Matching"):
            if not descriptions:
                st.warning("Please enter at least one description.")
            else:
                inputs = clip_processor(text=descriptions, images=image, return_tensors="pt", padding=True)
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # image-text similarity
                probs = logits_per_image.softmax(dim=1)[0]  # probabilities for each description

                scores = list(zip(descriptions, probs.tolist()))
                scores.sort(key=lambda x: x[1], reverse=True)
                st.session_state.clip_scores = scores
                st.session_state.clip_best = scores[0][0]
                st.success(f"Best CLIP Match: {scores[0][0]} ({scores[0][1]*100:.2f}%)")

    # Ask user question
    user_question = st.text_input("Ask a question about the image:")

    if st.button("Get Answer from Local LLM"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            if mode == "BLIP (Caption Generation)":
                if not st.session_state.blip_caption:
                    st.warning("Please generate BLIP caption first.")
                else:
                    prompt = f"Image Caption: {st.session_state.blip_caption}\nUser Question: {user_question}\nAnswer briefly:"
            else:  # CLIP
                if not st.session_state.clip_scores:
                    st.warning("Please run CLIP description matching first.")
                else:
                    context = "Image was compared to the following descriptions:\n"
                    for desc, prob in st.session_state.clip_scores:
                        context += f"- {desc}: {prob*100:.2f}%\n"
                    context += f"\nBest match: {st.session_state.clip_best}\n"
                    prompt = f"{context}User Question: {user_question}\nAnswer briefly:"

            if 'prompt' in locals():
                with st.spinner("LLM is thinking..."):
                    try:
                        response = ollama.chat(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are an expert image analyst."},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        answer = response['message']['content']
                    except Exception as e:
                        answer = f"Error: {e}"
                st.markdown(f"**LLM Answer:** {answer}")

else:
    st.info("Please upload an image to get started.")

# Add model info in sidebar
st.sidebar.markdown("### Models Used")
st.sidebar.markdown("- **BLIP**: Salesforce/blip-image-captioning-base")
st.sidebar.markdown("- **CLIP**: openai/clip-vit-base-patch32")
st.sidebar.markdown("- **LLM**: Local Ollama App")