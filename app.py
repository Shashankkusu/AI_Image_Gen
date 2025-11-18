import os
import torch
import base64
import io
import time
import threading
import sqlite3
import json
import csv
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import logging
import zipfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    HF_TOKEN = os.environ.get('HF_TOKEN', 'hf_*****************************')
    
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
    METADATA_FOLDER = "image_metadata"
    EXPORT_FOLDER = "exports"
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    CACHE_DIR = "./model_cache"
    DATABASE_PATH = "./ai_images.db"
    
    # Supported export formats
    SUPPORTED_FORMATS = ['PNG', 'JPEG', 'WEBP']
    DEFAULT_QUALITY = 95

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
                filepath TEXT NOT NULL,
                prompt TEXT NOT NULL,
                enhanced_prompt TEXT NOT NULL,
                model_used TEXT NOT NULL,
                style TEXT NOT NULL,
                negative_prompt TEXT,
                generation_time REAL NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                format_used TEXT NOT NULL,
                quality INTEGER NOT NULL,
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
        
        # Create exports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                export_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                item_count INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                filename, filepath, prompt, enhanced_prompt, model_used, style, 
                negative_prompt, generation_time, image_width, image_height, 
                file_size, format_used, quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_data['filename'],
            image_data['filepath'],
            image_data['prompt'],
            image_data['enhanced_prompt'],
            image_data['model_used'],
            image_data['style'],
            image_data.get('negative_prompt', ''),
            image_data['generation_time'],
            image_data['image_width'],
            image_data['image_height'],
            image_data['file_size'],
            image_data['format_used'],
            image_data['quality']
        ))
        
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        print(f"💾 Saved generation record #{record_id} to database")
        return record_id
        
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        return None

def save_export_record(export_data):
    """Save export record to database"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO exports (export_type, filename, filepath, file_size, item_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            export_data['export_type'],
            export_data['filename'],
            export_data['filepath'],
            export_data['file_size'],
            export_data['item_count']
        ))
        
        conn.commit()
        conn.close()
        print(f"💾 Saved export record for {export_data['filename']}")
        
    except Exception as e:
        print(f"❌ Error saving export record: {e}")

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
        
        # Total storage used
        cursor.execute('SELECT SUM(file_size) FROM generated_images')
        total_storage = cursor.fetchone()[0] or 0
        
        conn.close()
        
        stats = {
            'total_generations': total_generations,
            'avg_generation_time': round(avg_generation_time, 2) if avg_generation_time else 0,
            'recent_generations': recent_generations,
            'total_storage_mb': round(total_storage / (1024 * 1024), 2)
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
            SELECT filename, prompt, model_used, style, generation_time, created_at, file_size
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
            'created_at': row[5],
            'file_size_mb': round(row[6] / (1024 * 1024), 2)
        } for row in generations]
        
    except Exception as e:
        print(f"❌ Error getting recent generations: {e}")
        return []

def get_all_generations():
    """Get all generations for export"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM generated_images 
            ORDER BY created_at DESC
        ''')
        
        columns = [column[0] for column in cursor.description]
        generations = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        
        return generations
        
    except Exception as e:
        print(f"❌ Error getting all generations: {e}")
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
    watermark_text = "AI Generated - Magic Canvas Pro"
    
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

def save_image_metadata(image_data, filename):
    """Save image metadata to JSON file"""
    try:
        metadata = {
            'filename': filename,
            'prompt': image_data['prompt'],
            'enhanced_prompt': image_data['enhanced_prompt'],
            'model_used': image_data['model_used'],
            'style': image_data['style'],
            'negative_prompt': image_data.get('negative_prompt', ''),
            'generation_time': image_data['generation_time'],
            'image_width': image_data['image_width'],
            'image_height': image_data['image_height'],
            'file_size': image_data['file_size'],
            'format_used': image_data['format_used'],
            'quality': image_data['quality'],
            'created_at': datetime.now().isoformat()
        }
        
        metadata_filename = f"{Path(filename).stem}_metadata.json"
        metadata_path = os.path.join(Config.METADATA_FOLDER, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    except Exception as e:
        print(f"❌ Error saving metadata: {e}")
        return None

def create_export_zip(images_data, export_format, quality=95):
    """Create a ZIP file with exported images and metadata"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"ai_images_export_{timestamp}.zip"
        zip_path = os.path.join(Config.EXPORT_FOLDER, zip_filename)
        
        os.makedirs(Config.EXPORT_FOLDER, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add images
            for img_data in images_data:
                original_path = img_data['filepath']
                if os.path.exists(original_path):
                    # Convert to desired format if needed
                    if export_format.upper() != 'PNG':
                        img = Image.open(original_path)
                        converted_filename = f"{Path(img_data['filename']).stem}.{export_format.lower()}"
                        converted_path = os.path.join(Config.EXPORT_FOLDER, converted_filename)
                        
                        if export_format.upper() == 'JPEG':
                            img = img.convert('RGB')  # JPEG doesn't support transparency
                        
                        img.save(converted_path, format=export_format.upper(), quality=quality)
                        zipf.write(converted_path, converted_filename)
                        os.remove(converted_path)  # Clean up temporary file
                    else:
                        zipf.write(original_path, img_data['filename'])
            
            # Add metadata CSV
            metadata_csv = os.path.join(Config.EXPORT_FOLDER, f"metadata_{timestamp}.csv")
            with open(metadata_csv, 'w', newline='') as csvfile:
                fieldnames = ['filename', 'prompt', 'model_used', 'style', 'generation_time', 'created_at']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for img_data in images_data:
                    writer.writerow({
                        'filename': img_data['filename'],
                        'prompt': img_data['prompt'],
                        'model_used': img_data['model_used'],
                        'style': img_data['style'],
                        'generation_time': img_data['generation_time'],
                        'created_at': img_data['created_at']
                    })
            
            zipf.write(metadata_csv, f"metadata_{timestamp}.csv")
            os.remove(metadata_csv)  # Clean up temporary file
            
            # Add README
            readme_content = f"""AI Image Generator Export
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Images: {len(images_data)}
Export Format: {export_format}
Quality: {quality}

This archive contains:
- Generated images in {export_format} format
- Metadata CSV file with generation details

Created with Magic Canvas Pro AI Image Generator
"""
            readme_path = os.path.join(Config.EXPORT_FOLDER, "README.txt")
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            zipf.write(readme_path, "README.txt")
            os.remove(readme_path)
        
        file_size = os.path.getsize(zip_path)
        
        # Save export record
        export_data = {
            'export_type': f'zip_{export_format}',
            'filename': zip_filename,
            'filepath': zip_path,
            'file_size': file_size,
            'item_count': len(images_data)
        }
        save_export_record(export_data)
        
        return zip_path, zip_filename, file_size
        
    except Exception as e:
        print(f"❌ Error creating export ZIP: {e}")
        return None, None, 0

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
        export_format = data.get('format', 'PNG').upper()
        quality = data.get('quality', 95)
        
        # Validate input
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        if len(prompt) > 500:
            return jsonify({"error": "Prompt too long. Maximum 500 characters"}), 400
        
        if export_format not in Config.SUPPORTED_FORMATS:
            return jsonify({"error": f"Unsupported format. Choose from: {', '.join(Config.SUPPORTED_FORMATS)}"}), 400
        
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
        
        # Save image in specified format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_image_{timestamp}.{export_format.lower()}"
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
        if export_format == 'JPEG':
            image = image.convert('RGB')  # JPEG doesn't support transparency
        
        image.save(filepath, format=export_format, quality=quality)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Convert to base64 for response
        buffered = io.BytesIO()
        image.save(buffered, format=export_format, quality=quality)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare data for database
        image_data = {
            'filename': filename,
            'filepath': filepath,
            'prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'model_used': model_key,
            'style': style,
            'negative_prompt': negative_prompt,
            'generation_time': generation_time,
            'image_width': image.width,
            'image_height': image.height,
            'file_size': file_size,
            'format_used': export_format,
            'quality': quality
        }
        
        # Save to database
        record_id = save_generation_record(image_data)
        
        # Save metadata file
        metadata_path = save_image_metadata(image_data, filename)
        
        return jsonify({
            "success": True,
            "images": [{
                "data": f"data:image/{export_format.lower()};base64,{img_str}",
                "filename": filename,
                "prompt": prompt,
                "model": model_key,
                "style": style,
                "generation_time": round(generation_time, 2),
                "format": export_format,
                "quality": quality
            }],
            "generation_time": generation_time,
            "enhanced_prompt": enhanced_prompt,
            "model_used": model_key,
            "record_id": record_id,
            "metadata_saved": metadata_path is not None
        })
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/api/export', methods=['POST'])
def export_images():
    """Export multiple images with metadata"""
    try:
        data = request.json
        export_format = data.get('format', 'PNG').upper()
        quality = data.get('quality', 95)
        include_metadata = data.get('include_metadata', True)
        
        if export_format not in Config.SUPPORTED_FORMATS:
            return jsonify({"error": f"Unsupported format. Choose from: {', '.join(Config.SUPPORTED_FORMATS)}"}), 400
        
        # Get all generations for export
        generations = get_all_generations()
        
        if not generations:
            return jsonify({"error": "No images to export"}), 400
        
        # Create ZIP export
        zip_path, zip_filename, file_size = create_export_zip(generations, export_format, quality)
        
        if not zip_path:
            return jsonify({"error": "Failed to create export file"}), 500
        
        return jsonify({
            "success": True,
            "export_file": zip_filename,
            "file_size": file_size,
            "item_count": len(generations),
            "format": export_format
        })
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route('/api/export/<filename>')
def download_export(filename):
    """Download export file"""
    try:
        return send_file(
            os.path.join(Config.EXPORT_FOLDER, filename),
            as_attachment=True,
            download_name=filename
        )
    except FileNotFoundError:
        return jsonify({"error": "Export file not found"}), 404

@app.route('/api/images')
def list_images():
    """Get paginated list of all images"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM generated_images')
        total = cursor.fetchone()[0]
        
        # Get paginated results
        offset = (page - 1) * per_page
        cursor.execute('''
            SELECT filename, prompt, model_used, style, generation_time, created_at, file_size
            FROM generated_images 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        ''', (per_page, offset))
        
        images = cursor.fetchall()
        conn.close()
        
        result = {
            'images': [{
                'filename': row[0],
                'prompt': row[1],
                'model_used': row[2],
                'style': row[3],
                'generation_time': round(row[4], 2),
                'created_at': row[5],
                'file_size_mb': round(row[6] / (1024 * 1024), 2)
            } for row in images],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"List images error: {str(e)}")
        return jsonify({"error": f"Failed to list images: {str(e)}"}), 500

@app.route('/api/images/<filename>/metadata')
def get_image_metadata(filename):
    """Get metadata for a specific image"""
    try:
        metadata_filename = f"{Path(filename).stem}_metadata.json"
        metadata_path = os.path.join(Config.METADATA_FOLDER, metadata_filename)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return jsonify(metadata)
        else:
            return jsonify({"error": "Metadata not found"}), 404
            
    except Exception as e:
        logger.error(f"Metadata error: {str(e)}")
        return jsonify({"error": f"Failed to get metadata: {str(e)}"}), 500

@app.route('/api/system/cleanup', methods=['POST'])
def cleanup_system():
    """Clean up temporary files and optimize system"""
    try:
        data = request.json
        delete_old = data.get('delete_old', False)
        days_old = data.get('days_old', 30)
        
        cleanup_report = {
            'deleted_files': 0,
            'freed_space': 0,
            'errors': []
        }
        
        if delete_old:
            # Delete files older than specified days
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            
            for folder in [Config.UPLOAD_FOLDER, Config.EXPORT_FOLDER, Config.METADATA_FOLDER]:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        filepath = os.path.join(folder, filename)
                        if os.path.isfile(filepath):
                            if os.path.getctime(filepath) < cutoff_time:
                                try:
                                    file_size = os.path.getsize(filepath)
                                    os.remove(filepath)
                                    cleanup_report['deleted_files'] += 1
                                    cleanup_report['freed_space'] += file_size
                                except Exception as e:
                                    cleanup_report['errors'].append(f"Failed to delete {filename}: {str(e)}")
        
        cleanup_report['freed_space_mb'] = round(cleanup_report['freed_space'] / (1024 * 1024), 2)
        
        return jsonify({
            "success": True,
            "cleanup_report": cleanup_report
        })
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

@app.route('/api/system_info')
def system_info():
    stats = get_generation_stats()
    
    # Get folder sizes
    def get_folder_size(folder):
        if not os.path.exists(folder):
            return 0
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
    
    folder_sizes = {
        'images_mb': round(get_folder_size(Config.UPLOAD_FOLDER) / (1024 * 1024), 2),
        'exports_mb': round(get_folder_size(Config.EXPORT_FOLDER) / (1024 * 1024), 2),
        'metadata_mb': round(get_folder_size(Config.METADATA_FOLDER) / (1024 * 1024), 2),
        'cache_mb': round(get_folder_size(Config.CACHE_DIR) / (1024 * 1024), 2)
    }
    
    return jsonify({
        "device": "cpu",
        "loaded_models": list(loaded_models.keys()),
        "total_models": len(Config.MODELS),
        "cache_dir": Config.CACHE_DIR,
        "database_path": Config.DATABASE_PATH,
        "statistics": stats,
        "storage_usage": folder_sizes,
        "supported_formats": Config.SUPPORTED_FORMATS
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
    os.makedirs(Config.METADATA_FOLDER, exist_ok=True)
    os.makedirs(Config.EXPORT_FOLDER, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    # Initialize database
    init_database()
    
    print("=" * 70)
    print("🚀 AI Image Generator Pro - WITH STORAGE & EXPORT SYSTEM")
    print("=" * 70)
    print("📊 Enhanced Features:")
    print("   • Multi-format export (PNG, JPEG, WEBP)")
    print("   • Metadata storage (JSON + CSV)")
    print("   • Batch export with ZIP downloads")
    print("   • Storage management and cleanup")
    print("   • Image gallery with pagination")
    print("=" * 70)
    print("💡 Storage Features:")
    print("   • Organized folder structure")
    print("   • Metadata preservation")
    print("   • Export with quality control")
    print("   • System cleanup tools")
    print("=" * 70)
    print("🌐 Open http://localhost:5000 in your browser")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
