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
        const icon = theme === 'dark' ? 'bi-sun' : 'bi-moon';
        const text = theme === 'dark' ? 'Light Theme' : 'Dark Theme';
        toggleBtn.innerHTML = `<i class="bi ${icon}"></i> ${text}`;
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
        icon.classList.remove('bi-eye');
        icon.classList.add('bi-eye-slash');
    } else {
        field.type = 'password';
        icon.classList.remove('bi-eye-slash');
        icon.classList.add('bi-eye');
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

    // Handle completion
    if (data.status === 'completed') {
        window.illustratorApp.processing = false;
        showSuccess('Processing completed successfully!');

        // Redirect to gallery if available
        if (data.manuscript_id) {
            setTimeout(() => {
                window.location.href = `/manuscript/${data.manuscript_id}/gallery`;
            }, 2000);
        }
    }

    // Handle errors
    if (data.status === 'error') {
        window.illustratorApp.processing = false;
        showError(`Processing error: ${data.error || 'Unknown error'}`);
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