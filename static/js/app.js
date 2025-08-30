/**
 * Smart Review Guardian - Main JavaScript Application
 */

// Global application state
window.ReviewGuardian = {
    darkMode: false,
    stats: null,
    currentResults: null
};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    setupDarkMode();
    setupGlobalEventListeners();
    loadInitialStats();
    console.log('🛡️ Smart Review Guardian initialized');
}

/**
 * Setup dark mode functionality
 */
function setupDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const body = document.body;
    
    // Check for saved dark mode preference
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    if (isDarkMode) {
        enableDarkMode();
    }
    
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            if (body.classList.contains('dark-mode')) {
                disableDarkMode();
            } else {
                enableDarkMode();
            }
        });
    }
}

/**
 * Enable dark mode
 */
function enableDarkMode() {
    document.body.classList.add('dark-mode');
    ReviewGuardian.darkMode = true;
    localStorage.setItem('darkMode', 'true');
    
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        darkModeToggle.innerHTML = '<i class="fas fa-sun me-1"></i>Light Mode';
    }
}

/**
 * Disable dark mode
 */
function disableDarkMode() {
    document.body.classList.remove('dark-mode');
    ReviewGuardian.darkMode = false;
    localStorage.setItem('darkMode', 'false');
    
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        darkModeToggle.innerHTML = '<i class="fas fa-moon me-1"></i>Dark Mode';
    }
}

/**
 * Setup global event listeners
 */
function setupGlobalEventListeners() {
    // Handle API errors globally
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        showNotification('An error occurred. Please try again.', 'error');
    });
    
    // Add loading animations to all buttons
    setupButtonLoadingStates();
}

/**
 * Setup button loading states
 */
function setupButtonLoadingStates() {
    document.addEventListener('click', function(e) {
        if (e.target.matches('button[type="submit"], .btn-loading')) {
            const button = e.target;
            const originalHTML = button.innerHTML;
            
            // Add loading state
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
            
            // Remove loading state after a delay (will be overridden by actual completion)
            setTimeout(() => {
                button.disabled = false;
                button.innerHTML = originalHTML;
            }, 5000);
        }
    });
}

/**
 * Load initial statistics
 */
function loadInitialStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            ReviewGuardian.stats = data;
            updateStatsDisplay(data);
        })
        .catch(error => {
            console.warn('Could not load initial stats:', error);
        });
}

/**
 * Update statistics display throughout the app
 */
function updateStatsDisplay(stats) {
    // Update any stats elements that exist on the current page
    const elements = {
        'totalReviews': stats.total_processed,
        'genuineCount': stats.genuine_count,
        'flaggedCount': stats.total_processed - stats.genuine_count,
        'lastUpdated': `Last updated: ${formatDate(stats.last_updated)}`
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
}

/**
 * Format date for display
 */
function formatDate(dateString) {
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch (error) {
        return dateString || 'Never';
    }
}

/**
 * Show notification to user
 */
function showNotification(message, type = 'info', duration = 5000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 10000;
        min-width: 300px;
        max-width: 500px;
    `;
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

/**
 * Make API request with error handling
 */
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Get prediction badge HTML
 */
function getPredictionBadge(prediction, confidence = null) {
    const config = {
        'genuine': {
            class: 'bg-success',
            text: 'Genuine',
            icon: 'fas fa-check-circle'
        },
        'suspicious': {
            class: 'bg-warning',
            text: 'Suspicious',
            icon: 'fas fa-question-circle'
        },
        'low-quality': {
            class: 'bg-warning',
            text: 'Low Quality',
            icon: 'fas fa-exclamation-triangle'
        },
        'high-confidence-spam': {
            class: 'bg-danger',
            text: 'Spam',
            icon: 'fas fa-ban'
        }
    };
    
    const pred = config[prediction] || config['suspicious'];
    const confidenceText = confidence ? ` (${Math.round(confidence * 100)}%)` : '';
    
    return `
        <span class="badge ${pred.class}">
            <i class="${pred.icon} me-1"></i>
            ${pred.text}${confidenceText}
        </span>
    `;
}

/**
 * Get action badge HTML
 */
function getActionBadge(action) {
    const config = {
        'automatic-approval': {
            class: 'bg-success',
            text: 'Auto-Approve',
            icon: 'fas fa-check'
        },
        'requires-manual-verification': {
            class: 'bg-warning',
            text: 'Manual Review',
            icon: 'fas fa-eye'
        },
        'automatic-rejection': {
            class: 'bg-danger',
            text: 'Auto-Reject',
            icon: 'fas fa-times'
        }
    };
    
    const act = config[action] || config['requires-manual-verification'];
    
    return `
        <span class="badge ${act.class}">
            <i class="${act.icon} me-1"></i>
            ${act.text}
        </span>
    `;
}

/**
 * Create progress bar HTML
 */
function createProgressBar(value, max = 100, className = '') {
    const percentage = Math.round((value / max) * 100);
    return `
        <div class="progress ${className}" style="height: 8px;">
            <div class="progress-bar" 
                 role="progressbar" 
                 style="width: ${percentage}%" 
                 aria-valuenow="${value}" 
                 aria-valuemin="0" 
                 aria-valuemax="${max}">
            </div>
        </div>
    `;
}

/**
 * Debounce function for performance
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * Smooth scroll to element
 */
function scrollToElement(element, offset = 0) {
    if (typeof element === 'string') {
        element = document.getElementById(element) || document.querySelector(element);
    }
    
    if (element) {
        const top = element.offsetTop - offset;
        window.scrollTo({
            top: top,
            behavior: 'smooth'
        });
    }
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('Copied to clipboard!', 'success', 2000);
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        showNotification('Failed to copy to clipboard', 'error');
    }
}

/**
 * Download data as file
 */
function downloadAsFile(data, filename, type = 'text/plain') {
    const blob = new Blob([data], { type });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

/**
 * Validate file type and size
 */
function validateFile(file, allowedTypes = [], maxSize = 16 * 1024 * 1024) {
    const errors = [];
    
    if (allowedTypes.length > 0) {
        const fileType = file.type || '';
        const fileExt = file.name.split('.').pop().toLowerCase();
        
        const isValidType = allowedTypes.some(type => 
            fileType.includes(type) || type.includes(fileExt)
        );
        
        if (!isValidType) {
            errors.push(`Invalid file type. Allowed: ${allowedTypes.join(', ')}`);
        }
    }
    
    if (file.size > maxSize) {
        errors.push(`File too large. Maximum size: ${formatFileSize(maxSize)}`);
    }
    
    return {
        valid: errors.length === 0,
        errors
    };
}

/**
 * Initialize tooltips (if Bootstrap tooltips are used)
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Auto-refresh functionality
 */
function setupAutoRefresh(callback, interval = 30000) {
    let refreshInterval;
    
    function startRefresh() {
        refreshInterval = setInterval(callback, interval);
    }
    
    function stopRefresh() {
        if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }
    }
    
    // Start auto-refresh
    startRefresh();
    
    // Pause when page is not visible
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
            stopRefresh();
        } else {
            startRefresh();
        }
    });
    
    // Return control functions
    return { start: startRefresh, stop: stopRefresh };
}

// Export functions for global use
window.ReviewGuardian.utils = {
    showNotification,
    apiRequest,
    formatFileSize,
    getPredictionBadge,
    getActionBadge,
    createProgressBar,
    debounce,
    scrollToElement,
    copyToClipboard,
    downloadAsFile,
    validateFile,
    initializeTooltips,
    setupAutoRefresh,
    formatDate
};

console.log('📱 Smart Review Guardian utilities loaded');