/**
 * Manuscript Illustrator Web Interface - Main JavaScript
 */

// Global state management
window.illustratorApp = {
    apiKeys: {},
    currentTheme: 'light',
    websocket: null,
    processing: false
};

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Load saved preferences
    loadUserPreferences();

    // Setup event listeners
    setupEventListeners();

    // Initialize tooltips and popovers
    initializeBootstrapComponents();

    console.log('Manuscript Illustrator Web Interface initialized');
}

function loadUserPreferences() {
    // Load theme preference
    const savedTheme = localStorage.getItem('illustrator_theme');
    if (savedTheme) {
        setTheme(savedTheme);
    }

    // Load API keys from localStorage (not recommended for production)
    const savedKeys = localStorage.getItem('illustrator_api_keys');
    if (savedKeys) {
        try {
            window.illustratorApp.apiKeys = JSON.parse(savedKeys);
            populateApiKeyForm();
        } catch (error) {
            console.warn('Failed to load saved API keys:', error);
        }
    }
}

function setupEventListeners() {
    // Theme toggle
    document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);

    // Global keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

function initializeBootstrapComponents() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// Theme management
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-bs-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-bs-theme', theme);
    localStorage.setItem('illustrator_theme', theme);
    window.illustratorApp.currentTheme = theme;

    // Update theme toggle button text
    const toggleBtn = document.getElementById('themeToggle');
    if (toggleBtn) {
        const icon = theme === 'dark' ? 'brightness-high' : 'moon-stars';
        const text = theme === 'dark' ? 'Light Theme' : 'Dark Theme';
        toggleBtn.innerHTML = `<i class="bi bi-${icon} me-2"></i>${text}`;
    }
}

// Keyboard shortcuts
function handleKeyboardShortcuts(event) {
    // Ctrl/Cmd + N: New manuscript
    if ((event.ctrlKey || event.metaKey) && event.key === 'n') {
        event.preventDefault();
        window.location.href = '/manuscript/new';
    }

    // Escape: Close modals
    if (event.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const bootstrapModal = bootstrap.Modal.getInstance(modal);
            if (bootstrapModal) {
                bootstrapModal.hide();
            }
        });
    }
}

// API Key management
function saveApiKeys() {
    const apiKeys = {
        anthropic: document.getElementById('anthropicApiKey').value.trim(),
        openai: document.getElementById('openaiApiKey').value.trim(),
        huggingface: document.getElementById('huggingfaceApiKey').value.trim(),
        google_credentials: document.getElementById('googleCredentials').value.trim()
    };

    // Validate at least one key is provided
    const hasAnyKey = Object.values(apiKeys).some(key => key.length > 0);
    if (!hasAnyKey) {
        showError('Please provide at least one API key');
        return;
    }

    // Save to app state and localStorage
    window.illustratorApp.apiKeys = apiKeys;
    localStorage.setItem('illustrator_api_keys', JSON.stringify(apiKeys));

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('apiKeysModal'));
    modal.hide();

    showSuccess('API keys saved successfully');
}

function populateApiKeyForm() {
    const keys = window.illustratorApp.apiKeys;
    if (keys.anthropic) document.getElementById('anthropicApiKey').value = keys.anthropic;
    if (keys.openai) document.getElementById('openaiApiKey').value = keys.openai;
    if (keys.huggingface) document.getElementById('huggingfaceApiKey').value = keys.huggingface;
    if (keys.google_credentials) document.getElementById('googleCredentials').value = keys.google_credentials;
}

function togglePasswordVisibility(fieldId) {
    const field = document.getElementById(fieldId);
    const button = field.nextElementSibling;
    const icon = button.querySelector('i');

    if (field.type === 'password') {
        field.type = 'text';
        icon.className = 'bi bi-eye-slash';
    } else {
        field.type = 'password';
        icon.className = 'bi bi-eye';
    }
}

// Loading overlay management
function showLoadingOverlay(message = 'Loading...') {
    const overlay = document.getElementById('loadingOverlay');
    const messageEl = document.getElementById('loadingMessage');

    if (overlay && messageEl) {
        messageEl.textContent = message;
        overlay.classList.remove('d-none');
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.add('d-none');
    }
}

function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('d-none');
    }
}

function hideLoading() {
    const loadingElements = document.querySelectorAll('[id$="Loading"]');
    loadingElements.forEach(el => el.classList.add('d-none'));
}

function showEmpty(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('d-none');
    }
}

// Flash message system
function showSuccess(message, duration = 5000) {
    showFlashMessage(message, 'success', duration);
}

function showError(message, duration = 8000) {
    showFlashMessage(message, 'danger', duration);
}

function showWarning(message, duration = 6000) {
    showFlashMessage(message, 'warning', duration);
}

function showInfo(message, duration = 5000) {
    showFlashMessage(message, 'info', duration);
}

function showFlashMessage(message, type, duration) {
    const flashContainer = document.getElementById('flashMessages');
    if (!flashContainer) return;

    const alertId = 'alert_' + Date.now();
    const alertEl = document.createElement('div');
    alertEl.id = alertId;
    alertEl.className = `alert alert-${type} alert-dismissible fade show`;
    alertEl.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    flashContainer.appendChild(alertEl);

    // Auto-dismiss after duration
    if (duration > 0) {
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                const bootstrapAlert = new bootstrap.Alert(alert);
                bootstrapAlert.close();
            }
        }, duration);
    }
}

// WebSocket management
function connectWebSocket(sessionId) {
    if (window.illustratorApp.websocket) {
        window.illustratorApp.websocket.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/processing/${sessionId}`;

    window.illustratorApp.websocket = new WebSocket(wsUrl);

    window.illustratorApp.websocket.onopen = function(event) {
        console.log('WebSocket connected:', event);
        showInfo('Connected to processing server');
    };

    window.illustratorApp.websocket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (error) {
            console.log('WebSocket message:', event.data);
        }
    };

    window.illustratorApp.websocket.onclose = function(event) {
        console.log('WebSocket closed:', event);
        if (window.illustratorApp.processing) {
            showWarning('Connection to processing server lost. Attempting to reconnect...');
            // Attempt to reconnect after 5 seconds
            setTimeout(() => connectWebSocket(sessionId), 5000);
        }
    };

    window.illustratorApp.websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
        showError('Processing server connection error');
    };
}

function handleWebSocketMessage(data) {
    console.log('WebSocket data received:', data);

    // Handle processing status updates
    if (data.status) {
        updateProcessingStatus(data);
    }

    // Handle new images
    if (data.type === 'image' && data.image_url) {
        displayGeneratedImage(data.image_url, data.prompt || '');
    }

    // Handle completion
    if (data.status === 'completed' || data.type === 'complete') {
        window.illustratorApp.processing = false;
        showSuccess(`Processing completed successfully! Generated ${data.images_count || 0} images.`);

        // Redirect to gallery if available
        if (data.manuscript_id) {
            setTimeout(() => {
                window.location.href = `/manuscript/${data.manuscript_id}/gallery`;
            }, 2000);
        }
    }

    // Handle errors
    if (data.status === 'error' || data.type === 'error') {
        window.illustratorApp.processing = false;
        showError(`Processing error: ${data.error || 'Unknown error'}`);
    }

    // Handle progress updates
    if (data.type === 'progress' && data.progress !== undefined) {
        updateProgress(data.progress, data.message);
    }

    // Handle log messages
    if (data.type === 'log') {
        displayLogMessage(data);
    }

    // Handle step updates
    if (data.type === 'step') {
        updateStep(data.step, data.status);
    }
}

function updateProcessingStatus(data) {
    // Update progress bar if present
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar && data.progress !== undefined) {
        progressBar.style.width = `${data.progress}%`;
        progressBar.textContent = `${data.progress}%`;
    }

    // Update status message
    const statusEl = document.getElementById('processingStatus');
    if (statusEl && data.message) {
        statusEl.textContent = data.message;
    }

    // Update current chapter info
    const chapterEl = document.getElementById('currentChapter');
    if (chapterEl && data.current_chapter) {
        chapterEl.textContent = `Chapter ${data.current_chapter} of ${data.total_chapters}`;
    }
}

function displayGeneratedImage(imageUrl, prompt) {
    // Find or create images container
    let imagesContainer = document.getElementById('generatedImages');
    if (!imagesContainer) {
        // Create container if it doesn't exist
        const processingContainer = document.querySelector('.processing-container') || document.body;
        imagesContainer = document.createElement('div');
        imagesContainer.id = 'generatedImages';
        imagesContainer.className = 'generated-images-container mt-4';
        imagesContainer.innerHTML = '<h5>Generated Images</h5><div class="row" id="imageRow"></div>';
        processingContainer.appendChild(imagesContainer);
    }

    // Add new image to container
    const imageRow = document.getElementById('imageRow');
    if (imageRow) {
        const imageCol = document.createElement('div');
        imageCol.className = 'col-md-4 mb-3';
        imageCol.innerHTML = `
            <div class="card">
                <img src="${imageUrl}" class="card-img-top" alt="Generated Image"
                     onclick="openLightbox('${imageUrl}', '${escapeHtml(prompt)}')"
                     style="cursor: pointer; height: 200px; object-fit: cover;">
                <div class="card-body">
                    <p class="card-text small text-muted">${escapeHtml(prompt.substring(0, 100))}${prompt.length > 100 ? '...' : ''}</p>
                </div>
            </div>
        `;
        imageRow.appendChild(imageCol);

        // Show success message for image generation
        showSuccess('New image generated!', 3000);
    }
}

function updateProgress(progress, message) {
    // Update progress bar
    const progressBar = document.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }

    // Update status message
    const statusEl = document.getElementById('processingStatus');
    if (statusEl && message) {
        statusEl.textContent = message;
    }
}

function displayLogMessage(logData) {
    // Find or create log container
    let logContainer = document.getElementById('processingLog');
    if (!logContainer) {
        const processingContainer = document.querySelector('.processing-container') || document.body;
        logContainer = document.createElement('div');
        logContainer.id = 'processingLog';
        logContainer.className = 'processing-log mt-3';
        logContainer.innerHTML = '<h6>Processing Log</h6><div class="log-messages" style="height: 300px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 5px;"></div>';
        processingContainer.appendChild(logContainer);
    }

    const logMessages = document.querySelector('.log-messages');
    if (logMessages && logData.message) {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry text-${logData.level === 'error' ? 'danger' : logData.level === 'warning' ? 'warning' : logData.level === 'success' ? 'success' : 'muted'}`;
        logEntry.innerHTML = `<small>[${new Date().toLocaleTimeString()}] ${escapeHtml(logData.message)}</small>`;
        logMessages.appendChild(logEntry);

        // Auto-scroll to bottom
        logMessages.scrollTop = logMessages.scrollHeight;
    }
}

function updateStep(stepNumber, status) {
    // Update processing steps if they exist
    const stepEl = document.getElementById(`step-${stepNumber}`);
    if (stepEl) {
        stepEl.className = `step-item step-${status}`;
        const icon = stepEl.querySelector('.step-icon');
        if (icon) {
            if (status === 'completed') {
                icon.innerHTML = '<i class="bi bi-check-circle-fill text-success"></i>';
            } else if (status === 'processing') {
                icon.innerHTML = '<i class="bi bi-arrow-repeat text-primary"></i>';
            } else {
                icon.innerHTML = '<i class="bi bi-circle text-muted"></i>';
            }
        }
    }
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    } catch (error) {
        return dateString;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
}

// Image handling
function previewImage(file, previewElementId) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById(previewElementId);
        if (img) {
            img.src = e.target.result;
            img.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
}

function openLightbox(imageSrc, caption = '') {
    const lightbox = document.createElement('div');
    lightbox.className = 'lightbox-overlay';
    lightbox.innerHTML = `
        <div class="lightbox-content">
            <button class="lightbox-close" onclick="closeLightbox(this)">&times;</button>
            <img src="${imageSrc}" alt="${caption}" class="lightbox-image">
            ${caption ? `<div class="text-center mt-2 text-white">${caption}</div>` : ''}
        </div>
    `;

    document.body.appendChild(lightbox);

    // Close on click outside
    lightbox.addEventListener('click', function(e) {
        if (e.target === lightbox) {
            closeLightbox(lightbox);
        }
    });

    // Close on escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeLightbox(lightbox);
        }
    }, { once: true });
}

function closeLightbox(element) {
    const lightbox = element.closest ? element.closest('.lightbox-overlay') : element;
    if (lightbox && lightbox.parentNode) {
        lightbox.parentNode.removeChild(lightbox);
    }
}

// Export for global access
window.illustratorUtils = {
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showLoadingOverlay,
    hideLoadingOverlay,
    connectWebSocket,
    escapeHtml,
    formatDate,
    formatFileSize,
    debounce,
    openLightbox,
    closeLightbox
};