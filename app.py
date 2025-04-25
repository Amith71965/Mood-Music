from flask import Flask, render_template, request, jsonify
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model and processor
try:
    logger.info("Loading MusicGen model and processor...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    if torch.cuda.is_available():
        logger.info("CUDA is available, moving model to GPU...")
        model = model.to("cuda")
    else:
        logger.info("Running on CPU mode - generation might be slower")
    print("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    processor = None
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_music():
    if model is None or processor is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not initialized properly'
        }), 500
        
    try:
        description = request.json.get('description', '')
        duration = int(request.json.get('duration', 10))
        
        logger.info(f"Generating music for description: '{description}' with duration: {duration}s")
        
        # Process the text input
        logger.info("Processing text input...")
        inputs = processor(
            text=[description],
            padding=True,
            return_tensors="pt",
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate audio
        logger.info("Generating audio (this may take a while)...")
        audio_values = model.generate(
            **inputs,
            max_length=duration * 50,  # approximate mapping of seconds to tokens
            do_sample=True,
        )
        
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_music_{timestamp}.wav"
        filepath = os.path.join('static', 'generated', filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the audio file
        logger.info("Saving generated audio...")
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_values = audio_values.cpu()
        torchaudio.save(filepath, audio_values.squeeze(0), sample_rate=sampling_rate)
        
        logger.info("Generation completed successfully!")
        return jsonify({
            'status': 'success',
            'file_path': f'/static/generated/{filename}'
        })
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Create directory for generated files
    os.makedirs('static/generated', exist_ok=True)
    logger.info("Starting Flask server...")
    app.run(debug=True, port=8501) 