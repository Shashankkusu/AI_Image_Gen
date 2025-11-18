class MagicCanvasPro {
    constructor() {
        this.selectedModel = null;
        this.models = {};
        this.currentModalImage = null;
        this.currentPage = 1;
        this.totalPages = 1;
        this.initializeApp();
    }

    async initializeApp() {
        // Show intro animation for 3 seconds
        await this.showIntroAnimation();
        
        // Load models, stats and initialize the app
        await this.loadModels();
        await this.loadStats();
        await this.loadStorageInfo();
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

    async loadStorageInfo() {
        try {
            const response = await fetch('/api/system_info');
            const systemInfo = await response.json();
            this.updateStorageInfo(systemInfo);
        } catch (error) {
            console.error('Failed to load storage info:', error);
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
        document.getElementById('totalStorage').textContent = (stats.total_storage_mb || 0) + ' MB';
        
        if (stats.most_used_model) {
            // Update if you have an element for popular model
            const popularModelElement = document.getElementById('popularModel');
            if (popularModelElement) {
                popularModelElement.textContent = this.formatModelName(stats.most_used_model.model);
            }
        }
    }

    updateStorageInfo(systemInfo) {
        const storageInfo = document.getElementById('storageInfo');
        if (!storageInfo) return;

        const storage = systemInfo.storage_usage || {};
        
        storageInfo.innerHTML = `
            <div class="storage-item">
                <div class="storage-label">Generated Images</div>
                <div class="storage-value">${storage.images_mb || 0} MB</div>
            </div>
            <div class="storage-item">
                <div class="storage-label">Exports</div>
                <div class="storage-value">${storage.exports_mb || 0} MB</div>
            </div>
            <div class="storage-item">
                <div class="storage-label">Metadata</div>
                <div class="storage-value">${storage.metadata_mb || 0} MB</div>
            </div>
            <div class="storage-item">
                <div class="storage-label">Model Cache</div>
                <div class="storage-value">${storage.cache_mb || 0} MB</div>
            </div>
        `;
    }

    updateRecentGenerations(generations) {
        const container = document.getElementById('recentGenerationsList');
        if (!container) return;

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
                        ${this.formatModelName(gen.model_used)} • ${gen.style} • ${gen.generation_time}s • ${gen.file_size_mb}MB
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
        if (!modelGrid) return;

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

        if (progressBar) progressBar.style.width = `${progress.progress}%`;
        
        const statusMessages = {
            'starting': '🚀 Starting download...',
            'downloading': '📥 Downloading model files...',
            'optimizing': '⚡ Optimizing for your system...',
            'ready': '✅ Ready to generate! 🎉',
            'error': '❌ Download failed'
        };

        if (statusText) statusText.textContent = statusMessages[progress.status] || progress.status;
    }

    showDownloadSection() {
        const section = document.getElementById('downloadSection');
        if (section) section.classList.remove('hidden');
    }

    hideDownloadSection() {
        const section = document.getElementById('downloadSection');
        if (section) section.classList.add('hidden');
    }

    selectModel(modelKey) {
        this.selectedModel = modelKey;
        this.renderModelCards();
        this.updateGenerateButton();
        this.showSuccess(`${this.formatModelName(modelKey)} selected! Ready to create.`);
    }

    updateGenerateButton() {
        const generateBtn = document.getElementById('generateBtn');
        if (!generateBtn) return;
        
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
        
        if (promptTextarea && charCount) {
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
        }

        // Quality slider
        const qualitySlider = document.getElementById('quality');
        const qualityValue = document.getElementById('qualityValue');
        
        if (qualitySlider && qualityValue) {
            qualitySlider.addEventListener('input', () => {
                qualityValue.textContent = `${qualitySlider.value}%`;
            });
        }

        // Export quality slider
        const exportQualitySlider = document.getElementById('exportQuality');
        const exportQualityValue = document.getElementById('exportQualityValue');
        
        if (exportQualitySlider && exportQualityValue) {
            exportQualitySlider.addEventListener('input', () => {
                exportQualityValue.textContent = `${exportQualitySlider.value}%`;
            });
        }

        // Generate button
        const generateBtn = document.getElementById('generateBtn');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => {
                this.generateImage();
            });
        }

        // Enter key to generate
        if (promptTextarea) {
            promptTextarea.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    this.generateImage();
                }
            });
        }

        // ESC key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeModal();
                this.closeExportModal();
                this.closeGalleryModal();
                this.closeCleanupModal();
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
        const format = document.getElementById('format').value;
        const quality = document.getElementById('quality').value;

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
                    negative_prompt: negativePrompt,
                    format: format,
                    quality: parseInt(quality)
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
                await this.loadStorageInfo();
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

        if (!resultsSection || !generationInfo || !imagesGrid) return;

        // Show generation info
        generationInfo.innerHTML = `
            <div style="display: grid; gap: 8px;">
                <div><strong>🎯 Prompt:</strong> ${data.images[0].prompt}</div>
                <div><strong>🤖 Model:</strong> ${this.formatModelName(data.model_used)}</div>
                <div><strong>🎨 Style:</strong> ${document.getElementById('style').options[document.getElementById('style').selectedIndex].text}</div>
                <div><strong>📁 Format:</strong> ${data.images[0].format} (Quality: ${data.images[0].quality}%)</div>
                <div><strong>⏱️ Time:</strong> ${data.generation_time.toFixed(2)} seconds</div>
                <div><strong>✨ Enhanced:</strong> "${data.enhanced_prompt}"</div>
                ${data.record_id ? `<div><strong>📊 Record ID:</strong> #${data.record_id}</div>` : ''}
                ${data.metadata_saved ? `<div><strong>💾 Metadata:</strong> Saved with image</div>` : ''}
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
                <div style="font-weight: 500;">✨ Your AI Creation (${imageData.format})</div>
                <div class="action-buttons">
                    <button class="view-btn" onclick="magicCanvas.openModal('${imageData.data}', '${imageData.filename}')">
                        🔍 View Full Size
                    </button>
                    <a href="/api/download/${imageData.filename}" class="download-btn" download="${imageData.filename}">
                        💾 Download ${imageData.format}
                    </a>
                    <button class="download-btn" onclick="downloadImageDirect('${imageData.data}', '${imageData.filename}')">
                        ⬇️ Save As...
                    </button>
                </div>
            </div>
        `;
        
        return card;
    }

    // Storage and Export Methods
    async showExportModal() {
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            
            const exportInfo = document.getElementById('exportInfo');
            if (exportInfo) {
                exportInfo.innerHTML = `
                    <div class="export-stat">
                        <span>Total Images:</span>
                        <span>${stats.total_generations || 0}</span>
                    </div>
                    <div class="export-stat">
                        <span>Estimated Size:</span>
                        <span>${stats.total_storage_mb || 0} MB</span>
                    </div>
                    <div class="export-stat">
                        <span>Format:</span>
                        <span id="currentExportFormat">PNG</span>
                    </div>
                `;
            }
            
            const modal = document.getElementById('exportModal');
            modal.classList.add('active');
        } catch (error) {
            this.showError('Failed to load export information: ' + error.message);
        }
    }

    closeExportModal() {
        const modal = document.getElementById('exportModal');
        modal.classList.remove('active');
    }

    async startExport() {
        const format = document.getElementById('exportFormat').value;
        const quality = document.getElementById('exportQuality').value;

        try {
            const response = await fetch('/api/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    format: format,
                    quality: parseInt(quality),
                    include_metadata: true
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Export failed');
            }

            if (data.success) {
                this.showSuccess(`Export created successfully! Downloading ${data.export_file}...`);
                this.closeExportModal();
                
                // Download the file
                window.location.href = `/api/export/${data.export_file}`;
                
                // Refresh storage info
                await this.loadStorageInfo();
            }

        } catch (error) {
            this.showError('Export failed: ' + error.message);
        }
    }

    async showGallery() {
        await this.loadGalleryPage(1);
        const modal = document.getElementById('galleryModal');
        modal.classList.add('active');
    }

    closeGalleryModal() {
        const modal = document.getElementById('galleryModal');
        modal.classList.remove('active');
    }

    async loadGalleryPage(page) {
        try {
            const response = await fetch(`/api/images?page=${page}&per_page=20`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to load gallery');
            }

            this.currentPage = page;
            this.totalPages = data.pagination.pages;

            this.updateGalleryPagination();
            this.renderGallery(data.images);
        } catch (error) {
            this.showError('Failed to load gallery: ' + error.message);
        }
    }

    updateGalleryPagination() {
        const paginationInfo = document.getElementById('paginationInfo');
        const prevButton = document.getElementById('prevPage');
        const nextButton = document.getElementById('nextPage');

        if (paginationInfo) {
            paginationInfo.textContent = `Page ${this.currentPage} of ${this.totalPages}`;
        }

        if (prevButton) {
            prevButton.disabled = this.currentPage <= 1;
        }

        if (nextButton) {
            nextButton.disabled = this.currentPage >= this.totalPages;
        }
    }

    renderGallery(images) {
        const galleryGrid = document.getElementById('galleryGrid');
        if (!galleryGrid) return;

        galleryGrid.innerHTML = '';

        if (images.length === 0) {
            galleryGrid.innerHTML = '<div style="text-align: center; color: var(--text-secondary); padding: 40px;">No images found</div>';
            return;
        }

        images.forEach(image => {
            const item = document.createElement('div');
            item.className = 'gallery-item';
            item.innerHTML = `
                <img src="/api/download/${image.filename}" alt="${image.prompt}" class="gallery-thumbnail">
                <div class="gallery-info">
                    <div class="gallery-prompt">${image.prompt}</div>
                    <div class="gallery-meta">
                        ${this.formatModelName(image.model_used)} • ${image.generation_time}s • ${image.file_size_mb}MB
                    </div>
                </div>
            `;
            
            item.addEventListener('click', () => {
                this.openGalleryImage(image.filename);
            });
            
            galleryGrid.appendChild(item);
        });
    }

    async openGalleryImage(filename) {
        try {
            const response = await fetch(`/api/images/${filename}/metadata`);
            if (response.ok) {
                const metadata = await response.json();
                this.showImageMetadata(metadata);
            } else {
                // Fallback to just showing the image
                this.openModal(`/api/download/${filename}`, filename);
            }
        } catch (error) {
            this.openModal(`/api/download/${filename}`, filename);
        }
    }

    showImageMetadata(metadata) {
        const modal = document.createElement('div');
        modal.className = 'modal active';
        modal.innerHTML = `
            <div class="modal-content large">
                <button class="modal-close" onclick="this.parentElement.parentElement.remove()">×</button>
                <h2>📋 Image Metadata</h2>
                <div class="metadata-content">
                    <img src="/api/download/${metadata.filename}" alt="Preview" style="max-width: 100%; border-radius: 10px; margin-bottom: 20px;">
                    <div style="display: grid; gap: 10px;">
                        ${Object.entries(metadata).map(([key, value]) => `
                            <div><strong>${key}:</strong> ${value}</div>
                        `).join('')}
                    </div>
                </div>
                <div class="modal-actions">
                    <a href="/api/download/${metadata.filename}" class="download-btn" download="${metadata.filename}">
                        💾 Download Image
                    </a>
                    <button class="btn btn-secondary" onclick="this.parentElement.parentElement.parentElement.remove()">
                        Close
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    prevPage() {
        if (this.currentPage > 1) {
            this.loadGalleryPage(this.currentPage - 1);
        }
    }

    nextPage() {
        if (this.currentPage < this.totalPages) {
            this.loadGalleryPage(this.currentPage + 1);
        }
    }

    showCleanupModal() {
        const modal = document.getElementById('cleanupModal');
        modal.classList.add('active');
    }

    closeCleanupModal() {
        const modal = document.getElementById('cleanupModal');
        modal.classList.remove('active');
    }

    async startCleanup() {
        const deleteOldFiles = document.getElementById('deleteOldFiles').checked;
        const daysOld = document.getElementById('daysOld').value;

        try {
            const response = await fetch('/api/system/cleanup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    delete_old: deleteOldFiles,
                    days_old: parseInt(daysOld)
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Cleanup failed');
            }

            if (data.success) {
                const report = data.cleanup_report;
                this.showSuccess(`Cleanup completed! Deleted ${report.deleted_files} files, freed ${report.freed_space_mb} MB`);
                this.closeCleanupModal();
                
                // Refresh storage info
                await this.loadStorageInfo();
                await this.loadStats();
                
                if (report.errors.length > 0) {
                    console.warn('Cleanup warnings:', report.errors);
                }
            }

        } catch (error) {
            this.showError('Cleanup failed: ' + error.message);
        }
    }

    // Existing methods
    openModal(imageSrc, filename) {
        this.currentModalImage = { src: imageSrc, filename: filename };
        const modal = document.getElementById('imageModal');
        const modalImage = document.getElementById('modalImage');
        
        if (modal && modalImage) {
            modalImage.src = imageSrc;
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    }

    closeModal() {
        const modal = document.getElementById('imageModal');
        if (modal) {
            modal.classList.remove('active');
            document.body.style.overflow = 'auto';
            this.currentModalImage = null;
        }
    }

    downloadModalImage() {
        if (this.currentModalImage) {
            downloadImageDirect(this.currentModalImage.src, this.currentModalImage.filename);
        }
    }

    showError(message) {
        const errorSection = document.getElementById('errorSection');
        const errorMessage = document.getElementById('errorMessage');
        
        if (errorSection && errorMessage) {
            errorMessage.textContent = message;
            errorSection.classList.remove('hidden');
            errorSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Error: ' + message);
        }
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
        const errorSection = document.getElementById('errorSection');
        if (errorSection) errorSection.classList.add('hidden');
    }

    hideResults() {
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) resultsSection.classList.add('hidden');
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

function closeExportModal() {
    magicCanvas.closeExportModal();
}

function closeGalleryModal() {
    magicCanvas.closeGalleryModal();
}

function closeCleanupModal() {
    magicCanvas.closeCleanupModal();
}

function downloadModalImage() {
    magicCanvas.downloadModalImage();
}

// Close modals when clicking outside
document.addEventListener('click', function(e) {
    const modals = ['imageModal', 'exportModal', 'galleryModal', 'cleanupModal'];
    
    modals.forEach(modalId => {
        const modal = document.getElementById(modalId);
        if (modal && e.target === modal) {
            switch(modalId) {
                case 'imageModal': closeModal(); break;
                case 'exportModal': closeExportModal(); break;
                case 'galleryModal': closeGalleryModal(); break;
                case 'cleanupModal': closeCleanupModal(); break;
            }
        }
    });
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
