import os
import torch
import base64
import io
import time
import threading
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    HF_TOKEN = os.environ.get('HF_TOKEN', 'hf_**************************')
    
    # Available models (tested and working)
    MODELS = {
        "tiny": {
            "name": "OFA-Sys/small-stable-diffusion-v0",
            "size": "400MB",
            "speed": "Fast",
            "quality": "Good",
            "resolution": 384,
            "type": "stable_diffusion"
        },
        "small": {
            "name": "runwayml/stable-diffusion-v1-5", 
            "size": "7GB",
            "speed": "Medium",
            "quality": "Excellent",
            "resolution": 512,
            "type": "stable_diffusion"
        },
        "dreamshaper": {
            "name": "Lykon/DreamShaper",
            "size": "5GB",
            "speed": "Medium",
            "quality": "Excellent",
            "resolution": 512,
            "type": "stable_diffusion"
        },
        "openjourney": {
            "name": "prompthero/openjourney-v4",
            "size": "4GB",
            "speed": "Medium",
            "quality": "Very Good",
            "resolution": 512,
            "type": "stable_diffusion"
        },
        "mini-sd": {
            "name": "OFA-Sys/small-stable-diffusion-v0",
            "size": "400MB",
            "speed": "Very Fast",
            "quality": "Good",
            "resolution": 256,
            "type": "stable_diffusion"
        },
        "portrait": {
            "name": "wavymulder/portraitplus",
            "size": "2GB",
            "speed": "Fast",
            "quality": "Very Good",
            "resolution": 384,
            "type": "stable_diffusion"
        }
    }
    
    UPLOAD_FOLDER = "generated_images"
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    CACHE_DIR = "./model_cache"
    DATABASE_PATH = "./ai_images.db"

app.config.from_object(Config)

# Global variables
loaded_models = {}
download_progress = {}
current_downloads = {}

# Database setup
def init_database():
    """Initialize the SQLite database"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generated_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prompt TEXT NOT NULL,
                enhanced_prompt TEXT NOT NULL,
                model_used TEXT NOT NULL,
                style TEXT NOT NULL,
                negative_prompt TEXT,
                generation_time REAL NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create model_downloads table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_downloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_size TEXT NOT NULL,
                download_time REAL NOT NULL,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully!")
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

def save_generation_record(image_data):
    """Save generation record to database"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO generated_images (
                filename, prompt, enhanced_prompt, model_used, style, 
                negative_prompt, generation_time, image_width, image_height, file_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_data['filename'],
            image_data['prompt'],
            image_data['enhanced_prompt'],
            image_data['model_used'],
            image_data['style'],
            image_data.get('negative_prompt', ''),
            image_data['generation_time'],
            image_data['image_width'],
            image_data['image_height'],
            image_data['file_size']
        ))
        
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        print(f"💾 Saved generation record #{record_id} to database")
        return record_id
        
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        return None

def save_download_record(model_name, model_size, download_time):
    """Save model download record to database"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_downloads (model_name, model_size, download_time)
            VALUES (?, ?, ?)
        ''', (model_name, model_size, download_time))
        
        conn.commit()
        conn.close()
        print(f"💾 Saved download record for {model_name}")
        
    except Exception as e:
        print(f"❌ Error saving download record: {e}")

def get_generation_stats():
    """Get generation statistics from database"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Total generations
        cursor.execute('SELECT COUNT(*) FROM generated_images')
        total_generations = cursor.fetchone()[0]
        
        # Most used model
        cursor.execute('''
            SELECT model_used, COUNT(*) as count 
            FROM generated_images 
            GROUP BY model_used 
            ORDER BY count DESC 
            LIMIT 1
        ''')
        most_used_model = cursor.fetchone()
        
        # Average generation time
        cursor.execute('SELECT AVG(generation_time) FROM generated_images')
        avg_generation_time = cursor.fetchone()[0]
        
        # Recent generations (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM generated_images 
            WHERE created_at >= datetime('now', '-1 day')
        ''')
        recent_generations = cursor.fetchone()[0]
        
        conn.close()
        
        stats = {
            'total_generations': total_generations,
            'avg_generation_time': round(avg_generation_time, 2) if avg_generation_time else 0,
            'recent_generations': recent_generations
        }
        
        if most_used_model:
            stats['most_used_model'] = {
                'model': most_used_model[0],
                'count': most_used_model[1]
            }
            
        return stats
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return {}

def get_recent_generations(limit=10):
    """Get recent generations for display"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, prompt, model_used, style, generation_time, created_at
            FROM generated_images 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        generations = cursor.fetchall()
        conn.close()
        
        return [{
            'filename': row[0],
            'prompt': row[1][:100] + '...' if len(row[1]) > 100 else row[1],
            'model_used': row[2],
            'style': row[3],
            'generation_time': round(row[4], 2),
            'created_at': row[5]
        } for row in generations]
        
    except Exception as e:
        print(f"❌ Error getting recent generations: {e}")
        return []

# Initialize database on startup
init_database()

def get_model_info(model_key):
    """Get model information"""
    return Config.MODELS.get(model_key, Config.MODELS["tiny"])

def download_model(model_key):
    """Download model in background thread"""
    if model_key in current_downloads and current_downloads[model_key]:
        return  # Already downloading
    
    current_downloads[model_key] = True
    download_progress[model_key] = {"status": "starting", "progress": 0}
    download_start_time = time.time()
    
    def download_thread():
        try:
            model_info = get_model_info(model_key)
            print(f"📥 Downloading {model_key} model ({model_info['size']})...")
            print(f"🔗 Model: {model_info['name']}")
            
            download_progress[model_key] = {"status": "downloading", "progress": 10}
            
            # Download the model with proper authentication
            token = Config.HF_TOKEN if Config.HF_TOKEN != 'hf_your_token_here' else None
            
            # Use safer loading method
            pipe = StableDiffusionPipeline.from_pretrained(
                model_info["name"],
                torch_dtype=torch.float32,
                cache_dir=Config.CACHE_DIR,
                token=token,
                safety_checker=None,
                requires_safety_checker=False,
                local_files_only=False,
                resume_download=True
            )
            
            download_progress[model_key] = {"status": "optimizing", "progress": 80}
            
            # Optimize for CPU
            pipe = pipe.to("cpu")
            pipe.enable_attention_slicing()
            
            # For smaller models, use fewer inference steps
            if model_key in ["tiny", "mini-sd"]:
                pipe.unet.set_attention_slice(1)
            
            loaded_models[model_key] = pipe
            
            # Calculate download time and save record
            download_time = time.time() - download_start_time
            save_download_record(model_key, model_info['size'], download_time)
            
            download_progress[model_key] = {"status": "ready", "progress": 100}
            current_downloads[model_key] = False
            
            print(f"✅ {model_key} model loaded successfully!")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error downloading {model_key} model: {error_msg}")
            
            # Provide helpful error messages
            if "401" in error_msg:
                error_msg = "Authentication failed. Please check your Hugging Face token."
            elif "403" in error_msg:
                error_msg = "Access denied. The model may require accepting terms on Hugging Face."
            elif "404" in error_msg:
                error_msg = "Model not found. The model name may have changed."
            elif "timed out" in error_msg.lower():
                error_msg = "Download timed out. Please check your internet connection."
            
            download_progress[model_key] = {"status": "error", "progress": 0, "error": error_msg}
            current_downloads[model_key] = False
    
    thread = threading.Thread(target=download_thread)
    thread.daemon = True
    thread.start()

def add_watermark(image):
    """Add watermark to image"""
    draw = ImageDraw.Draw(image)
    watermark_text = "AI Generated"
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    
    margin = 10
    x = image.width - text_width - margin
    y = image.height - 30
    
    draw.rectangle([x-5, y-5, x+text_width+5, y+20], fill=(0, 0, 0, 180))
    draw.text((x, y), watermark_text, fill=(255, 255, 255), font=font)
    
    return image

def enhance_prompt(prompt, style, model_key):
    """Enhance prompt with style descriptors"""
    style_descriptors = {
        "photorealistic": "photorealistic, highly detailed, professional photography, 4K, ultra-realistic",
        "artistic": "artistic, painterly, creative, expressive, masterpiece, oil painting",
        "cartoon": "cartoon style, animated, vibrant colors, clean lines, digital art",
        "anime": "anime style, Japanese animation, vibrant, detailed eyes, manga style",
        "digital_art": "digital art, concept art, trending on artstation, detailed, unreal engine",
        "fantasy": "fantasy, epic, magical, mystical, dreamlike, ethereal",
        "sci-fi": "sci-fi, futuristic, cyberpunk, neon, technological, advanced"
    }
    
    base_enhancements = "high quality, detailed, professional"
    style_enhancement = style_descriptors.get(style, "")
    
    # Model-specific enhancements
    if model_key == "portrait":
        base_enhancements = "portrait, professional photography, sharp focus, detailed face, perfect lighting"
        if not style_enhancement:
            style_enhancement = "photorealistic, professional portrait"
    elif model_key == "mini-sd":
        base_enhancements = "vibrant colors, artistic, creative"
    
    enhanced_prompt = f"{prompt}, {base_enhancements}"
    if style_enhancement:
        enhanced_prompt += f", {style_enhancement}"
    
    return enhanced_prompt

def filter_inappropriate_content(prompt):
    """Basic content filtering"""
    inappropriate_keywords = [
        "nude", "naked", "explicit", "porn", "sexual", "violence", 
        "hate", "racist", "offensive", "illegal", "nsfw"
    ]
    
    prompt_lower = prompt.lower()
    for keyword in inappropriate_keywords:
        if keyword in prompt_lower:
            return False, f"Prompt contains inappropriate content"
    
    return True, "Prompt is appropriate"

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get generation statistics"""
    stats = get_generation_stats()
    return jsonify(stats)

@app.route('/api/recent_generations')
def get_recent():
    """Get recent generations"""
    limit = request.args.get('limit', 10, type=int)
    generations = get_recent_generations(limit)
    return jsonify(generations)

@app.route('/api/models')
def get_models():
    """Get available models information"""
    models_info = {}
    for key, info in Config.MODELS.items():
        models_info[key] = {
            "name": info["name"],
            "size": info["size"],
            "speed": info["speed"],
            "quality": info["quality"],
            "resolution": info["resolution"],
            "loaded": key in loaded_models,
            "downloading": current_downloads.get(key, False)
        }
    return jsonify(models_info)

@app.route('/api/download/<model_key>', methods=['POST'])
def start_download(model_key):
    """Start downloading a specific model"""
    if model_key not in Config.MODELS:
        return jsonify({"error": "Invalid model"}), 400
    
    if model_key in loaded_models:
        return jsonify({"status": "already_loaded"})
    
    download_model(model_key)
    return jsonify({"status": "started"})

@app.route('/api/download_progress/<model_key>')
def get_download_progress(model_key):
    """Get download progress for a model"""
    progress = download_progress.get(model_key, {"status": "unknown", "progress": 0})
    return jsonify(progress)

@app.route('/api/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        model_key = data.get('model', 'tiny')
        style = data.get('style', 'photorealistic')
        negative_prompt = data.get('negative_prompt', '')
        
        # Validate input
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        if len(prompt) > 500:
            return jsonify({"error": "Prompt too long. Maximum 500 characters"}), 400
        
        # Check if model is loaded
        if model_key not in loaded_models:
            return jsonify({"error": f"Model {model_key} not loaded. Please download it first."}), 400
        
        # Content filtering
        is_appropriate, filter_message = filter_inappropriate_content(prompt)
        if not is_appropriate:
            return jsonify({"error": filter_message}), 400
        
        # Enhance prompt
        enhanced_prompt = enhance_prompt(prompt, style, model_key)
        
        # Default negative prompt
        if not negative_prompt:
            negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, worst quality, jpeg artifacts"
        
        model_info = get_model_info(model_key)
        pipe = loaded_models[model_key]
        
        print(f"🎨 Generating with {model_key} model...")
        start_time = time.time()
        
        # Generate image with model-specific settings
        images = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=20 if model_key in ["tiny", "mini-sd"] else 25,
            guidance_scale=7.0 if model_key in ["tiny", "mini-sd"] else 7.5,
            width=model_info["resolution"],
            height=model_info["resolution"]
        ).images
        
        generation_time = time.time() - start_time
        
        # Process image
        image = images[0]
        image = add_watermark(image)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_image_{timestamp}.png"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        image.save(filepath, "PNG")
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Prepare data for database
        image_data = {
            'filename': filename,
            'prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'model_used': model_key,
            'style': style,
            'negative_prompt': negative_prompt,
            'generation_time': generation_time,
            'image_width': image.width,
            'image_height': image.height,
            'file_size': file_size
        }
        
        # Save to database
        record_id = save_generation_record(image_data)
        
        return jsonify({
            "success": True,
            "images": [{
                "data": f"data:image/png;base64,{img_str}",
                "filename": filename,
                "prompt": prompt,
                "model": model_key,
                "style": style,
                "generation_time": round(generation_time, 2)
            }],
            "generation_time": generation_time,
            "enhanced_prompt": enhanced_prompt,
            "model_used": model_key,
            "record_id": record_id
        })
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/api/system_info')
def system_info():
    stats = get_generation_stats()
    return jsonify({
        "device": "cpu",
        "loaded_models": list(loaded_models.keys()),
        "total_models": len(Config.MODELS),
        "cache_dir": Config.CACHE_DIR,
        "database_path": Config.DATABASE_PATH,
        "statistics": stats
    })

@app.route('/api/download/<filename>')
def download_image(filename):
    try:
        return send_file(
            os.path.join(Config.UPLOAD_FOLDER, filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Create directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    # Initialize database
    init_database()
    
    print("=" * 70)
    print("🚀 AI Image Generator Pro - WITH DATABASE & ANALYTICS")
    print("=" * 70)
    print("📊 New Features:")
    print("   • SQLite database for storing generation history")
    print("   • Real-time statistics dashboard")
    print("   • Recent generations tracking")
    print("   • Model download analytics")
    print("=" * 70)
    print("💡 Database Features:")
    print("   • Store prompts, models, styles, and generation times")
    print("   • Track file sizes and image dimensions")
    print("   • View statistics and analytics")
    print("=" * 70)
    print("🌐 Open http://localhost:5000 in your browser")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=False)