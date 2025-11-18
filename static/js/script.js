class MagicCanvasPro {
    constructor() {
        this.selectedModel = null;
        this.models = {};
        this.currentModalImage = null;
        this.initializeApp();
    }

    async initializeApp() {
        // Show intro animation for 3 seconds
        await this.showIntroAnimation();
        
        // Load models, stats and initialize the app
        await this.loadModels();
        await this.loadStats();
        await this.loadRecentGenerations();
        this.initializeEventListeners();
        this.showMainApp();
    }

    async showIntroAnimation() {
        return new Promise(resolve => {
            setTimeout(() => {
                // Hide intro screen with animation
                const introScreen = document.getElementById('introScreen');
                introScreen.style.opacity = '0';
                introScreen.style.transform = 'scale(1.1)';
                
                setTimeout(() => {
                    introScreen.classList.add('hidden');
                    resolve();
                }, 800);
            }, 3000);
        });
    }

    showMainApp() {
        const mainApp = document.getElementById('mainApp');
        mainApp.classList.add('loaded');
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            this.models = await response.json();
            this.renderModelCards();
        } catch (error) {
            this.showError('Failed to load models: ' + error.message);
        }
    }

    async loadStats() {
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            this.updateStats(stats);
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }

    async loadRecentGenerations() {
        try {
            const response = await fetch('/api/recent_generations?limit=5');
            const generations = await response.json();
            this.updateRecentGenerations(generations);
        } catch (error) {
            console.error('Failed to load recent generations:', error);
        }
    }

    updateStats(stats) {
        document.getElementById('totalGenerations').textContent = stats.total_generations || 0;
        document.getElementById('avgGenerationTime').textContent = stats.avg_generation_time ? stats.avg_generation_time + 's' : '0s';
        document.getElementById('recentGenerations').textContent = stats.recent_generations || 0;
        
        if (stats.most_used_model) {
            document.getElementById('popularModel').textContent = this.formatModelName(stats.most_used_model.model);
        }
    }

    updateRecentGenerations(generations) {
        const container = document.getElementById('recentGenerationsList');
        container.innerHTML = '';

        if (generations.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 20px;">No generations yet. Create your first image!</div>';
            return;
        }

        generations.forEach(gen => {
            const item = document.createElement('div');
            item.className = 'generation-item';
            item.innerHTML = `
                <div class="generation-info-small">
                    <div class="generation-prompt">${gen.prompt}</div>
                    <div class="generation-meta">
                        ${this.formatModelName(gen.model_used)} • ${gen.style} • ${gen.generation_time}s
                    </div>
                </div>
                <div class="generation-meta">
                    ${new Date(gen.created_at).toLocaleDateString()}
                </div>
            `;
            container.appendChild(item);
        });
    }

    renderModelCards() {
        const modelGrid = document.getElementById('modelGrid');
        modelGrid.innerHTML = '';

        Object.entries(this.models).forEach(([key, model]) => {
            const modelCard = this.createModelCard(key, model);
            modelGrid.appendChild(modelCard);
        });
    }

    createModelCard(key, model) {
        const card = document.createElement('div');
        card.className = `model-card ${this.selectedModel === key ? 'selected' : ''}`;
        
        card.innerHTML = `
            <div class="model-header">
                <div class="model-name">${this.formatModelName(key)}</div>
                <div class="model-badge" style="background: ${this.getModelColor(key)}">
                    ${model.speed}
                </div>
            </div>
            <div class="model-details">
                <div class="model-detail">
                    <span class="detail-icon">📦</span>
                    <span>${model.size}</span>
                </div>
                <div class="model-detail">
                    <span class="detail-icon">🌟</span>
                    <span>${model.quality}</span>
                </div>
                <div class="model-detail">
                    <span class="detail-icon">🖼️</span>
                    <span>${model.resolution}px</span>
                </div>
                <div class="model-detail">
                    <span class="detail-icon">⚡</span>
                    <span>${model.speed}</span>
                </div>
            </div>
            <div class="model-actions">
                ${model.loaded ? 
                    `<button class="btn btn-primary" onclick="magicCanvas.selectModel('${key}')">
                        🎯 Select Model
                    </button>
                    <button class="btn btn-success" disabled>✅ Ready</button>` :
                    `<button class="btn btn-primary" onclick="magicCanvas.downloadModel('${key}')">
                        📥 Download ${model.size}
                    </button>`
                }
            </div>
        `;

        return card;
    }

    formatModelName(key) {
        const names = {
            'tiny': 'Small Model (Fast)',
            'small': 'Stable Diffusion v1.5',
            'dreamshaper': 'DreamShaper',
            'openjourney': 'OpenJourney v4',
            'mini-sd': 'Mini SD (Fastest)',
            'portrait': 'Portrait+ (People)'
        };
        return names[key] || key;
    }

    getModelColor(key) {
        const colors = {
            'tiny': '#10b981',
            'small': '#f59e0b',
            'dreamshaper': '#ec4899',
            'openjourney': '#8b5cf6',
            'mini-sd': '#06b6d4',
            'portrait': '#f97316'
        };
        return colors[key] || '#6366f1';
    }

    async downloadModel(modelKey) {
        this.showDownloadSection();
        
        try {
            const response = await fetch(`/api/download/${modelKey}`, {
                method: 'POST'
            });
            const data = await response.json();

            if (data.status === 'already_loaded') {
                this.models[modelKey].loaded = true;
                this.renderModelCards();
                this.hideDownloadSection();
                return;
            }

            this.pollDownloadProgress(modelKey);
            
        } catch (error) {
            this.showError('Failed to start download: ' + error.message);
        }
    }

    async pollDownloadProgress(modelKey) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`/api/download_progress/${modelKey}`);
                const progress = await response.json();

                this.updateDownloadProgress(progress);

                if (progress.status === 'ready') {
                    clearInterval(interval);
                    this.models[modelKey].loaded = true;
                    this.renderModelCards();
                    this.hideDownloadSection();
                    this.showSuccess('Model downloaded successfully! 🎉');
                } else if (progress.status === 'error') {
                    clearInterval(interval);
                    this.showError('Download failed: ' + progress.error);
                }
            } catch (error) {
                console.error('Progress check error:', error);
            }
        }, 1000);
    }

    updateDownloadProgress(progress) {
        const progressBar = document.getElementById('downloadProgress');
        const statusText = document.getElementById('downloadStatus');

        progressBar.style.width = `${progress.progress}%`;
        
        const statusMessages = {
            'starting': '🚀 Starting download...',
            'downloading': '📥 Downloading model files...',
            'optimizing': '⚡ Optimizing for your system...',
            'ready': '✅ Ready to generate! 🎉',
            'error': '❌ Download failed'
        };

        statusText.textContent = statusMessages[progress.status] || progress.status;
    }

    showDownloadSection() {
        document.getElementById('downloadSection').classList.remove('hidden');
    }

    hideDownloadSection() {
        document.getElementById('downloadSection').classList.add('hidden');
    }

    selectModel(modelKey) {
        this.selectedModel = modelKey;
        this.renderModelCards();
        this.updateGenerateButton();
        this.showSuccess(`${this.formatModelName(modelKey)} selected! Ready to create.`);
    }

    updateGenerateButton() {
        const generateBtn = document.getElementById('generateBtn');
        
        if (this.selectedModel) {
            generateBtn.disabled = false;
            generateBtn.innerHTML = `🎨 Generate with ${this.formatModelName(this.selectedModel)}`;
        } else {
            generateBtn.disabled = true;
            generateBtn.innerHTML = '🎯 Select a Model First to Generate';
        }
    }

    initializeEventListeners() {
        // Prompt character counter
        const promptTextarea = document.getElementById('prompt');
        const charCount = document.getElementById('charCount');
        
        promptTextarea.addEventListener('input', () => {
            const length = promptTextarea.value.length;
            charCount.textContent = length;
            
            if (length > 450) {
                charCount.style.color = '#ef4444';
            } else if (length > 350) {
                charCount.style.color = '#f59e0b';
            } else {
                charCount.style.color = '#cbd5e1';
            }
        });

        // Generate button
        document.getElementById('generateBtn').addEventListener('click', () => {
            this.generateImage();
        });

        // Enter key to generate
        promptTextarea.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.generateImage();
            }
        });

        // ESC key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });
    }

    async generateImage() {
        if (!this.selectedModel) {
            this.showError('Please select a model first');
            return;
        }

        const prompt = document.getElementById('prompt').value.trim();
        const style = document.getElementById('style').value;
        const negativePrompt = document.getElementById('negativePrompt').value;

        if (!prompt) {
            this.showError('Please enter a prompt description');
            return;
        }

        const generateBtn = document.getElementById('generateBtn');
        generateBtn.disabled = true;
        generateBtn.innerHTML = '✨ Generating... Please wait';
        this.hideError();
        this.hideResults();

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    model: this.selectedModel,
                    style: style,
                    negative_prompt: negativePrompt
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Generation failed');
            }

            if (data.success) {
                this.displayResults(data);
                // Refresh stats and recent generations after successful generation
                await this.loadStats();
                await this.loadRecentGenerations();
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }

        } catch (error) {
            this.showError(error.message);
        } finally {
            generateBtn.disabled = false;
            this.updateGenerateButton();
        }
    }

    displayResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const generationInfo = document.getElementById('generationInfo');
        const imagesGrid = document.getElementById('imagesGrid');

        // Show generation info
        generationInfo.innerHTML = `
            <div style="display: grid; gap: 8px;">
                <div><strong>🎯 Prompt:</strong> ${data.images[0].prompt}</div>
                <div><strong>🤖 Model:</strong> ${this.formatModelName(data.model_used)}</div>
                <div><strong>🎨 Style:</strong> ${document.getElementById('style').options[document.getElementById('style').selectedIndex].text}</div>
                <div><strong>⏱️ Time:</strong> ${data.generation_time.toFixed(2)} seconds</div>
                <div><strong>✨ Enhanced:</strong> "${data.enhanced_prompt}"</div>
                ${data.record_id ? `<div><strong>📊 Record ID:</strong> #${data.record_id}</div>` : ''}
            </div>
        `;

        // Display images
        imagesGrid.innerHTML = '';
        data.images.forEach((imageData, index) => {
            const imageCard = this.createImageCard(imageData, index);
            imagesGrid.appendChild(imageCard);
        });

        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    createImageCard(imageData, index) {
        const card = document.createElement('div');
        card.className = 'image-card';
        
        card.innerHTML = `
            <div class="image-wrapper">
                <img src="${imageData.data}" alt="Generated image" class="generated-image" 
                     onclick="magicCanvas.openModal('${imageData.data}', '${imageData.filename}')">
            </div>
            <div class="image-actions">
                <div style="font-weight: 500;">✨ Your AI Creation</div>
                <div class="action-buttons">
                    <button class="view-btn" onclick="magicCanvas.openModal('${imageData.data}', '${imageData.filename}')">
                        🔍 View Full Size
                    </button>
                    <a href="/api/download/${imageData.filename}" class="download-btn" download="${imageData.filename}">
                        💾 Download PNG
                    </a>
                    <button class="download-btn" onclick="downloadImageDirect('${imageData.data}', '${imageData.filename}')">
                        ⬇️ Save As...
                    </button>
                </div>
            </div>
        `;
        
        return card;
    }

    openModal(imageSrc, filename) {
        this.currentModalImage = { src: imageSrc, filename: filename };
        const modal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modalImage');
        
        modalImage.src = imageSrc;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    closeModal() {
        const modal = document.getElementById('imageModal');
        modal.classList.remove('active');
        document.body.style.overflow = 'auto';
        this.currentModalImage = null;
    }

    downloadModalImage() {
        if (this.currentModalImage) {
            downloadImageDirect(this.currentModalImage.src, this.currentModalImage.filename);
        }
    }

    showError(message) {
        const errorSection = document.getElementById('errorSection');
        const errorMessage = document.getElementById('errorMessage');
        
        errorMessage.textContent = message;
        errorSection.classList.remove('hidden');
        errorSection.scrollIntoView({ behavior: 'smooth' });
    }

    showSuccess(message) {
        // Create a temporary success notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--success);
            color: white;
            padding: 15px 25px;
            border-radius: 10px;
            z-index: 1000;
            animation: fadeIn 0.5s ease-out;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    hideError() {
        document.getElementById('errorSection').classList.add('hidden');
    }

    hideResults() {
        document.getElementById('resultsSection').classList.add('hidden');
    }
}

// Global functions
function downloadImageDirect(imageData, filename) {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename || 'ai-generated-image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function closeModal() {
    magicCanvas.closeModal();
}

function downloadModalImage() {
    magicCanvas.downloadModalImage();
}

// Close modal when clicking outside
document.getElementById('imageModal').addEventListener('click', function(e) {
    if (e.target === this) {
        closeModal();
    }
});

// Initialize the application
let magicCanvas;

document.addEventListener('DOMContentLoaded', () => {
    magicCanvas = new MagicCanvasPro();
});

// Test functions
window.testPrompt = function() {
    const prompts = [
        "A majestic dragon flying over a medieval castle at sunset with vibrant colors, highly detailed fantasy art, cinematic lighting",
        "A cyberpunk cityscape with neon lights, flying cars, and towering skyscrapers at night, futuristic architecture",
        "A serene landscape with mountains, a crystal clear lake, and golden hour lighting, photorealistic nature photography",
        "A cute robot gardening in a lush greenhouse with colorful flowers, cartoon style, vibrant and cheerful",
        "A beautiful portrait of a woman with flowing red hair in a field of wildflowers, professional photography",
        "Abstract geometric patterns with vibrant colors and dynamic shapes, modern art"
    ];
    const randomPrompt = prompts[Math.floor(Math.random() * prompts.length)];
    document.getElementById('prompt').value = randomPrompt;
    document.getElementById('prompt').dispatchEvent(new Event('input'));
};