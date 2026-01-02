/**
 * Crypto MoE Scanner - Frontend Application
 * Handles API communication, real-time updates, and table management
 */

// ============================================
// State Management
// ============================================
const state = {
    isScanning: false,
    results: {},
    pollInterval: null,
    sortColumn: 'prediction',
    sortDirection: 'desc',
    filter: 'all',
    searchQuery: ''
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    timeframeSelect: document.getElementById('timeframeSelect'),
    scanBtn: document.getElementById('scanBtn'),
    stopBtn: document.getElementById('stopBtn'),
    statusBadge: document.getElementById('statusBadge'),
    progressSection: document.getElementById('progressSection'),
    progressPair: document.getElementById('progressPair'),
    progressCount: document.getElementById('progressCount'),
    progressBar: document.getElementById('progressBar'),
    progressPercent: document.getElementById('progressPercent'),
    progressResults: document.getElementById('progressResults'),
    bullishCount: document.getElementById('bullishCount'),
    bearishCount: document.getElementById('bearishCount'),
    totalScanned: document.getElementById('totalScanned'),
    errorCount: document.getElementById('errorCount'),
    resultsBody: document.getElementById('resultsBody'),
    searchInput: document.getElementById('searchInput'),
    filterSelect: document.getElementById('filterSelect')
};

// ============================================
// API Functions
// ============================================
const api = {
    async startScan(timeframe) {
        const response = await fetch('/api/scan/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ timeframe })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start scan');
        }
        return response.json();
    },

    async stopScan() {
        const response = await fetch('/api/scan/stop', { method: 'POST' });
        return response.json();
    },

    async getStatus() {
        const response = await fetch('/api/scan/status');
        return response.json();
    },

    async getResults() {
        const response = await fetch('/api/scan/results');
        return response.json();
    }
};

// ============================================
// UI Update Functions
// ============================================
function updateStatus(text, isScanning = false) {
    const statusText = elements.statusBadge.querySelector('.status-text');
    statusText.textContent = text;
    
    if (isScanning) {
        elements.statusBadge.classList.add('scanning');
    } else {
        elements.statusBadge.classList.remove('scanning');
    }
}

function updateProgress(status) {
    elements.progressPair.textContent = status.current_pair || 'Scanning...';
    elements.progressCount.textContent = `${status.scanned_count} / ${status.total_pairs}`;
    elements.progressBar.style.width = `${status.progress}%`;
    elements.progressPercent.textContent = `${status.progress}%`;
    elements.progressResults.textContent = `${status.results_count} results`;
}

function updateStats(results) {
    const values = Object.values(results);
    const bullish = values.filter(r => r.prediction > 0).length;
    const bearish = values.filter(r => r.prediction < 0).length;
    
    elements.bullishCount.textContent = bullish;
    elements.bearishCount.textContent = bearish;
    elements.totalScanned.textContent = values.length;
}

function formatPrice(price) {
    if (!price) return '-';
    if (price >= 1000) return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (price >= 1) return price.toFixed(4);
    return price.toFixed(6);
}

function formatVolume(volume) {
    if (!volume) return '-';
    if (volume >= 1e9) return (volume / 1e9).toFixed(2) + 'B';
    if (volume >= 1e6) return (volume / 1e6).toFixed(2) + 'M';
    if (volume >= 1e3) return (volume / 1e3).toFixed(2) + 'K';
    return volume.toFixed(0);
}

function getSignalClass(prediction) {
    if (prediction > 0.5) return 'bullish';
    if (prediction < -0.5) return 'bearish';
    return 'neutral';
}

function getSignalText(prediction) {
    if (prediction > 1.5) return '🚀 Strong Buy';
    if (prediction > 0.5) return '📈 Buy';
    if (prediction < -1.5) return '💥 Strong Sell';
    if (prediction < -0.5) return '📉 Sell';
    return '➖ Neutral';
}

function getRsiClass(rsi) {
    if (rsi >= 70) return 'overbought';
    if (rsi <= 30) return 'oversold';
    return '';
}

function renderResults(results) {
    // Apply filters
    let filtered = Object.entries(results);
    
    // Search filter
    if (state.searchQuery) {
        const query = state.searchQuery.toLowerCase();
        filtered = filtered.filter(([pair]) => pair.toLowerCase().includes(query));
    }
    
    // Signal filter
    if (state.filter === 'bullish') {
        filtered = filtered.filter(([, data]) => data.prediction > 0);
    } else if (state.filter === 'bearish') {
        filtered = filtered.filter(([, data]) => data.prediction < 0);
    }
    
    // Sort
    filtered.sort((a, b) => {
        const [, dataA] = a;
        const [, dataB] = b;
        
        let valA, valB;
        switch (state.sortColumn) {
            case 'pair':
                valA = a[0];
                valB = b[0];
                break;
            case 'prediction':
                valA = dataA.prediction || 0;
                valB = dataB.prediction || 0;
                break;
            case 'price':
                valA = dataA.price || 0;
                valB = dataB.price || 0;
                break;
            case 'rsi':
                valA = dataA.rsi || 0;
                valB = dataB.rsi || 0;
                break;
            case 'volume':
                valA = dataA.volume || 0;
                valB = dataB.volume || 0;
                break;
            default:
                valA = dataA.prediction || 0;
                valB = dataB.prediction || 0;
        }
        
        if (typeof valA === 'string') {
            return state.sortDirection === 'asc' 
                ? valA.localeCompare(valB) 
                : valB.localeCompare(valA);
        }
        
        return state.sortDirection === 'asc' ? valA - valB : valB - valA;
    });
    
    // Render
    if (filtered.length === 0) {
        elements.resultsBody.innerHTML = `
            <tr class="empty-row">
                <td colspan="6">
                    <div class="empty-state">
                        <span class="empty-icon">🔍</span>
                        <p>No results match your filters.</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }
    
    const rows = filtered.map(([pair, data], index) => {
        const signalClass = getSignalClass(data.prediction);
        const predictionClass = data.prediction > 0 ? 'positive' : data.prediction < 0 ? 'negative' : '';
        const rsiClass = getRsiClass(data.rsi);
        
        // Clean pair name for display
        const displayPair = pair.replace(':USDT', '');
        
        return `
            <tr style="animation-delay: ${index * 0.02}s">
                <td class="pair-cell">${displayPair}</td>
                <td class="prediction-cell ${predictionClass}">${data.prediction.toFixed(4)}%</td>
                <td class="price-cell">$${formatPrice(data.price)}</td>
                <td class="rsi-cell ${rsiClass}">${data.rsi ? data.rsi.toFixed(1) : '-'}</td>
                <td class="volume-cell">${formatVolume(data.volume)}</td>
                <td><span class="signal-badge ${signalClass}">${getSignalText(data.prediction)}</span></td>
            </tr>
        `;
    }).join('');
    
    elements.resultsBody.innerHTML = rows;
}

// ============================================
// Scan Control Functions
// ============================================
async function startScan() {
    const timeframe = elements.timeframeSelect.value;
    
    try {
        await api.startScan(timeframe);
        
        state.isScanning = true;
        state.results = {};
        
        // Update UI
        elements.scanBtn.disabled = true;
        elements.stopBtn.disabled = false;
        elements.progressSection.style.display = 'block';
        updateStatus('Scanning...', true);
        
        // Start polling
        startPolling();
        
    } catch (error) {
        alert('Failed to start scan: ' + error.message);
    }
}

async function stopScan() {
    try {
        await api.stopScan();
        state.isScanning = false;
        stopPolling();
        
        // Update UI
        elements.scanBtn.disabled = false;
        elements.stopBtn.disabled = true;
        updateStatus('Stopped', false);
        
    } catch (error) {
        console.error('Failed to stop scan:', error);
    }
}

async function pollStatus() {
    try {
        const [status, resultsData] = await Promise.all([
            api.getStatus(),
            api.getResults()
        ]);
        
        // Update progress
        updateProgress(status);
        elements.errorCount.textContent = status.errors_count;
        
        // Update results
        state.results = resultsData.results;
        updateStats(state.results);
        renderResults(state.results);
        
        // Check if scan complete
        if (!status.is_scanning && state.isScanning) {
            state.isScanning = false;
            stopPolling();
            
            elements.scanBtn.disabled = false;
            elements.stopBtn.disabled = true;
            updateStatus('Complete', false);
            
            // Keep progress visible but update text
            elements.progressPair.textContent = 'Scan Complete!';
        }
        
    } catch (error) {
        console.error('Polling error:', error);
    }
}

function startPolling() {
    if (state.pollInterval) return;
    state.pollInterval = setInterval(pollStatus, 1000);
    pollStatus(); // Initial poll
}

function stopPolling() {
    if (state.pollInterval) {
        clearInterval(state.pollInterval);
        state.pollInterval = null;
    }
}

// ============================================
// Sorting Functions
// ============================================
function handleSort(column) {
    if (state.sortColumn === column) {
        state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        state.sortColumn = column;
        state.sortDirection = 'desc';
    }
    
    // Update header styles
    document.querySelectorAll('.results-table th.sortable').forEach(th => {
        th.classList.remove('sorted', 'asc');
        if (th.dataset.sort === state.sortColumn) {
            th.classList.add('sorted');
            if (state.sortDirection === 'asc') {
                th.classList.add('asc');
            }
        }
    });
    
    renderResults(state.results);
}

// ============================================
// Event Listeners
// ============================================
function initEventListeners() {
    // Scan controls
    elements.scanBtn.addEventListener('click', startScan);
    elements.stopBtn.addEventListener('click', stopScan);
    
    // Search
    elements.searchInput.addEventListener('input', (e) => {
        state.searchQuery = e.target.value;
        renderResults(state.results);
    });
    
    // Filter
    elements.filterSelect.addEventListener('change', (e) => {
        state.filter = e.target.value;
        renderResults(state.results);
    });
    
    // Sorting
    document.querySelectorAll('.results-table th.sortable').forEach(th => {
        th.addEventListener('click', () => handleSort(th.dataset.sort));
    });
}

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    
    // Set initial sort indicator
    const predictionHeader = document.querySelector('[data-sort="prediction"]');
    if (predictionHeader) {
        predictionHeader.classList.add('sorted');
    }
    
    console.log('🧠 Crypto MoE Scanner initialized');
});
