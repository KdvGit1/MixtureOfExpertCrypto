/**
 * Crypto MoE Scanner - Frontend Application
 * Handles API communication, real-time updates, and dual table management
 */

// ============================================
// State Management
// ============================================
const state = {
    isScanning: false,
    isContinuousMode: false,
    scanCycle: 0,
    results: {},
    pollInterval: null,
    longSortColumn: 'prediction',
    longSortDirection: 'desc',
    shortSortColumn: 'prediction',
    shortSortDirection: 'asc',
    searchQuery: ''
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    timeframeSelect: document.getElementById('timeframeSelect'),
    continuousToggle: document.getElementById('continuousToggle'),
    scanBtn: document.getElementById('scanBtn'),
    stopBtn: document.getElementById('stopBtn'),
    statusBadge: document.getElementById('statusBadge'),
    progressSection: document.getElementById('progressSection'),
    progressPair: document.getElementById('progressPair'),
    progressCount: document.getElementById('progressCount'),
    progressBar: document.getElementById('progressBar'),
    progressPercent: document.getElementById('progressPercent'),
    progressResults: document.getElementById('progressResults'),
    scanCycle: document.getElementById('scanCycle'),
    bullishCount: document.getElementById('bullishCount'),
    bearishCount: document.getElementById('bearishCount'),
    riskCount: document.getElementById('riskCount'),
    totalScanned: document.getElementById('totalScanned'),
    errorCount: document.getElementById('errorCount'),
    longResultsBody: document.getElementById('longResultsBody'),
    shortResultsBody: document.getElementById('shortResultsBody'),
    longTableCount: document.getElementById('longTableCount'),
    shortTableCount: document.getElementById('shortTableCount'),
    searchInput: document.getElementById('searchInput')
};

// ============================================
// API Functions
// ============================================
const api = {
    async startScan(timeframe, continuous = false) {
        const response = await fetch('/api/scan/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ timeframe, continuous })
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

    // Show cycle number if in continuous mode
    if (status.continuous_mode && status.scan_cycle > 0) {
        elements.scanCycle.textContent = `Cycle #${status.scan_cycle}`;
        elements.scanCycle.style.display = 'inline';
    } else {
        elements.scanCycle.style.display = 'none';
    }
}

function updateStats(results) {
    const values = Object.values(results);
    const bullish = values.filter(r => r.prediction > 0).length;
    const bearish = values.filter(r => r.prediction < 0).length;
    const risky = values.filter(r => r.is_high_risk).length;

    elements.bullishCount.textContent = bullish;
    elements.bearishCount.textContent = bearish;
    elements.riskCount.textContent = risky;
    elements.totalScanned.textContent = values.length;
    elements.longTableCount.textContent = bullish;
    elements.shortTableCount.textContent = bearish;
}

function formatPrice(price) {
    if (!price) return '-';
    if (price >= 1000) return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (price >= 1) return price.toFixed(4);
    return price.toFixed(6);
}

function formatTime(timeStr) {
    if (!timeStr) return '-';
    const parts = timeStr.split(' ');
    return parts.length > 1 ? parts[1] : timeStr;
}

function getRsiClass(rsi) {
    if (rsi >= 70) return 'overbought';
    if (rsi <= 30) return 'oversold';
    return '';
}

function getConfidenceClass(confidence) {
    if (confidence >= 80) return 'high-confidence';
    if (confidence >= 50) return 'medium-confidence';
    return 'low-confidence';
}

function createRiskTooltip(riskReasons) {
    if (!riskReasons || riskReasons.length === 0) return '';
    return riskReasons.join('\n');
}

function renderTable(tableBody, data, sortColumn, sortDirection, isLong) {
    // Apply search filter
    let filtered = data;
    if (state.searchQuery) {
        const query = state.searchQuery.toLowerCase();
        filtered = filtered.filter(([pair]) => pair.toLowerCase().includes(query));
    }

    // Sort
    filtered.sort((a, b) => {
        const [, dataA] = a;
        const [, dataB] = b;

        let valA, valB;
        switch (sortColumn) {
            case 'pair':
                valA = a[0];
                valB = b[0];
                break;
            case 'prediction':
                valA = Math.abs(dataA.prediction || 0);
                valB = Math.abs(dataB.prediction || 0);
                break;
            case 'price':
                valA = dataA.price || 0;
                valB = dataB.price || 0;
                break;
            case 'rsi':
                valA = dataA.rsi || 0;
                valB = dataB.rsi || 0;
                break;
            case 'confidence':
                valA = dataA.confidence || 0;
                valB = dataB.confidence || 0;
                break;
            case 'time':
                valA = dataA.updated_at || '';
                valB = dataB.updated_at || '';
                break;
            default:
                valA = Math.abs(dataA.prediction || 0);
                valB = Math.abs(dataB.prediction || 0);
        }

        if (typeof valA === 'string') {
            return sortDirection === 'asc'
                ? valA.localeCompare(valB)
                : valB.localeCompare(valA);
        }

        return sortDirection === 'asc' ? valA - valB : valB - valA;
    });

    // Render
    if (filtered.length === 0) {
        const emoji = isLong ? '📈' : '📉';
        const text = isLong ? 'No long signals' : 'No short signals';
        tableBody.innerHTML = `
            <tr class="empty-row">
                <td colspan="7">
                    <div class="empty-state">
                        <span class="empty-icon">${emoji}</span>
                        <p>${text}</p>
                    </div>
                </td>
            </tr>
        `;
        return;
    }

    const rows = filtered.map(([pair, data], index) => {
        const predictionClass = isLong ? 'positive' : 'negative';
        const rsiClass = getRsiClass(data.rsi);
        const displayPair = pair.replace(':USDT', '').replace('/USDT', '');

        // Risk indicator
        let riskCell = '<td class="risk-cell safe">✓</td>';
        if (data.is_high_risk) {
            const tooltip = createRiskTooltip(data.risk_reasons);
            riskCell = `<td class="risk-cell risky" title="${tooltip}">⚠️</td>`;
        }

        return `
            <tr class="${data.is_high_risk ? 'high-risk-row' : ''}" style="animation-delay: ${index * 0.02}s">
                ${riskCell}
                <td class="pair-cell">${displayPair}</td>
                <td class="prediction-cell ${predictionClass}">${data.prediction.toFixed(4)}%</td>
                <td class="confidence-cell ${getConfidenceClass(data.confidence)}">${data.confidence ? data.confidence.toFixed(0) + '%' : '-'}</td>
                <td class="price-cell">$${formatPrice(data.price)}</td>
                <td class="rsi-cell ${rsiClass}">${data.rsi ? data.rsi.toFixed(1) : '-'}</td>
                <td class="time-cell">${formatTime(data.updated_at)}</td>
            </tr>
        `;
    }).join('');

    tableBody.innerHTML = rows;
}

function renderResults(results) {
    const entries = Object.entries(results);

    // Split into long and short
    const longSignals = entries.filter(([, data]) => data.prediction > 0);
    const shortSignals = entries.filter(([, data]) => data.prediction < 0);

    // Render both tables
    renderTable(
        elements.longResultsBody,
        longSignals,
        state.longSortColumn,
        state.longSortDirection,
        true
    );

    renderTable(
        elements.shortResultsBody,
        shortSignals,
        state.shortSortColumn,
        state.shortSortDirection,
        false
    );
}

// ============================================
// Scan Control Functions
// ============================================
async function startScan() {
    const timeframe = elements.timeframeSelect.value;
    const continuous = elements.continuousToggle.checked;

    try {
        await api.startScan(timeframe, continuous);

        state.isScanning = true;
        state.isContinuousMode = continuous;
        state.results = {};

        // Update UI
        elements.scanBtn.disabled = true;
        elements.stopBtn.disabled = false;
        elements.timeframeSelect.disabled = true;
        elements.continuousToggle.disabled = true;
        elements.progressSection.style.display = 'block';

        const modeText = continuous ? 'Continuous Scanning...' : 'Scanning...';
        updateStatus(modeText, true);

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
        state.isContinuousMode = false;
        stopPolling();

        // Update UI
        elements.scanBtn.disabled = false;
        elements.stopBtn.disabled = true;
        elements.timeframeSelect.disabled = false;
        elements.continuousToggle.disabled = false;
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

        // Check if scan complete (only for non-continuous mode)
        if (!status.is_scanning && state.isScanning && !status.continuous_mode) {
            state.isScanning = false;
            state.isContinuousMode = false;
            stopPolling();

            elements.scanBtn.disabled = false;
            elements.stopBtn.disabled = true;
            elements.timeframeSelect.disabled = false;
            elements.continuousToggle.disabled = false;
            updateStatus('Complete', false);

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
function handleSort(column, table) {
    if (table === 'long') {
        if (state.longSortColumn === column) {
            state.longSortDirection = state.longSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            state.longSortColumn = column;
            state.longSortDirection = 'desc';
        }
    } else {
        if (state.shortSortColumn === column) {
            state.shortSortDirection = state.shortSortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            state.shortSortColumn = column;
            state.shortSortDirection = 'desc';
        }
    }

    // Update header styles
    const tableId = table === 'long' ? 'longTable' : 'shortTable';
    document.querySelectorAll(`#${tableId} th.sortable`).forEach(th => {
        th.classList.remove('sorted', 'asc');
        const sortCol = table === 'long' ? state.longSortColumn : state.shortSortColumn;
        const sortDir = table === 'long' ? state.longSortDirection : state.shortSortDirection;
        if (th.dataset.sort === sortCol) {
            th.classList.add('sorted');
            if (sortDir === 'asc') {
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

    // Sorting for both tables
    document.querySelectorAll('.results-table th.sortable').forEach(th => {
        th.addEventListener('click', () => handleSort(th.dataset.sort, th.dataset.table));
    });
}

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();

    // Set initial sort indicators
    const longPredHeader = document.querySelector('#longTable [data-sort="prediction"]');
    if (longPredHeader) {
        longPredHeader.classList.add('sorted');
    }

    const shortPredHeader = document.querySelector('#shortTable [data-sort="prediction"]');
    if (shortPredHeader) {
        shortPredHeader.classList.add('sorted');
    }

    console.log('🧠 Crypto MoE Scanner initialized with dual tables');
});
