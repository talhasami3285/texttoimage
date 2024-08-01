from flask import Flask, request, jsonify, send_file
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
from PIL import Image

app = Flask(__name__)

class CFG:
    device = "cuda"  # Use "cuda" if GPU is available and compatible
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "CompVis/stable-diffusion-v1-4"  # Updated model ID
    hf_token = "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"  # Replace with your Hugging Face token

# Load the stable-diffusion model with Hugging Face token
pipe = StableDiffusionPipeline.from_pretrained(CFG.image_gen_model_id, use_auth_token=CFG.hf_token)
pipe = pipe.to(CFG.device)  # Move to device (CPU or GPU)

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt', "A beautiful sunset over a mountain range")
    
    try:
        # Generate the image
        with torch.autocast(CFG.device):
            image = pipe(prompt, generator=CFG.generator, num_inference_steps=CFG.image_gen_steps).images[0]

        # Save image to a BytesIO object
        img_bytes = BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Return the image as a response
        return send_file(img_bytes, mimetype='image/png', as_attachment=True, attachment_filename='generated_image.png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
