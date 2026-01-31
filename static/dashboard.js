/**
 * ============================================
 * CRYPTO AI DASHBOARD - Investor Edition v3
 * Featured Models, Theme Toggle, Currency Selector
 * ============================================
 */

// State
let liveData = [];
let backtestData = [];
let selectedModel = null;
let priceChart = null;
let currentCurrency = 'USD';
let currencyRates = { USD: 1, EUR: 0.92, TRY: 35.5, GBP: 0.79 };

// DOM Elements
const modelList = document.getElementById('modelList');
const detailPanel = document.getElementById('detailPanel');
const tfFilter = document.getElementById('tfFilter');
const featuredModels = document.getElementById('featuredModels');

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initCurrency();
    loadLivePredictions();
    loadBacktestRankings();

    tfFilter.addEventListener('change', () => renderModelList(filterModels()));

    // Auto refresh every 60 seconds
    setInterval(loadLivePredictions, 60000);

    // First visit disclaimer
    if (!localStorage.getItem('disclaimerAccepted')) {
        setTimeout(() => showFullDisclaimer(), 1000);
    }
});

// ============================================
// THEME TOGGLE
// ============================================

function initTheme() {
    const saved = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeIcon(saved);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const newTheme = current === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const icon = document.querySelector('.theme-icon');
    if (icon) {
        icon.textContent = theme === 'dark' ? '🌙' : '☀️';
    }
}

// ============================================
// CURRENCY SELECTOR
// ============================================

function initCurrency() {
    const select = document.getElementById('currencySelect');
    if (select) {
        select.addEventListener('change', (e) => {
            currentCurrency = e.target.value;
            // Re-render with new currency
            renderModelList(filterModels());
            renderFeaturedModels();
        });
    }
}

function formatPrice(usdPrice) {
    if (!usdPrice) return 'N/A';
    const converted = usdPrice * currencyRates[currentCurrency];
    const symbol = { USD: '$', EUR: '€', TRY: '₺', GBP: '£' }[currentCurrency];
    return symbol + converted.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ============================================
// DISCLAIMER
// ============================================

function closeDisclaimer() {
    document.getElementById('disclaimerBanner').style.display = 'none';
}

function showFullDisclaimer() {
    document.getElementById('disclaimerModal').classList.add('active');
}

function closeModal() {
    document.getElementById('disclaimerModal').classList.remove('active');
    localStorage.setItem('disclaimerAccepted', 'true');
}

// ============================================
// API CALLS
// ============================================

async function loadLivePredictions() {
    try {
        modelList.innerHTML = `<div class="loading-state"><div class="spinner"></div><p>Loading...</p></div>`;

        const response = await fetch('/api/live-predictions');
        const data = await response.json();
        liveData = data.predictions;

        renderFeaturedModels();
        renderModelList(filterModels());
    } catch (error) {
        console.error('Error:', error);
        modelList.innerHTML = `<div class="loading-state"><p>❌ Error loading data</p></div>`;
    }
}

async function loadModelDetail(coin, timeframe) {
    try {
        detailPanel.innerHTML = `<div class="loading-state"><div class="spinner"></div><p>Loading...</p></div>`;

        const [analysisRes, backtestRes, chartRes] = await Promise.all([
            fetch(`/api/models/${coin}/${timeframe}`),
            fetch(`/api/backtest/${coin}/${timeframe}`),
            fetch(`/api/chart/${coin}/${timeframe}?limit=100`)
        ]);

        const analysisData = await analysisRes.json();
        const backtestData = backtestRes.ok ? await backtestRes.json() : null;
        const chartData = chartRes.ok ? await chartRes.json() : null;

        const modelInfo = liveData.find(m => m.coin === coin && m.timeframe === timeframe);

        renderModelDetail(analysisData, backtestData, chartData, modelInfo);
    } catch (error) {
        console.error('Error:', error);
        detailPanel.innerHTML = `<div class="empty-detail"><h3>Error</h3><p>${error.message}</p></div>`;
    }
}

// ============================================
// RENDERING
// ============================================

function filterModels() {
    const tf = tfFilter.value;
    return tf === 'all' ? liveData : liveData.filter(m => m.timeframe === tf);
}

function getFeaturedModels() {
    // Get ALL PRO models (accuracy >= 52%)
    return liveData.filter(m => m.accuracy >= 52);
}

// ============================================
// CAROUSEL SCROLL
// ============================================

function scrollFeatured(direction) {
    const container = document.getElementById('featuredModels');
    if (!container) return;

    const cardWidth = 300;
    const scrollAmount = cardWidth * 2;

    container.scrollBy({
        left: scrollAmount * direction,
        behavior: 'smooth'
    });

    // Update indicators after scroll
    setTimeout(updateScrollIndicators, 300);
}

function updateScrollIndicators() {
    const container = document.getElementById('featuredModels');
    const wrapper = document.getElementById('carouselWrapper');
    const leftBtn = document.getElementById('scrollLeft');
    const rightBtn = document.getElementById('scrollRight');

    if (!container || !wrapper) return;

    const scrollLeft = container.scrollLeft;
    const scrollWidth = container.scrollWidth;
    const clientWidth = container.clientWidth;
    const maxScroll = scrollWidth - clientWidth;

    // Update fade edges
    if (scrollLeft <= 10) {
        wrapper.className = 'featured-carousel-wrapper scrolled-start';
    } else if (scrollLeft >= maxScroll - 10) {
        wrapper.className = 'featured-carousel-wrapper scrolled-end';
    } else {
        wrapper.className = 'featured-carousel-wrapper scrolled-middle';
    }

    // Update arrow buttons
    if (leftBtn) leftBtn.disabled = scrollLeft <= 10;
    if (rightBtn) rightBtn.disabled = scrollLeft >= maxScroll - 10;

    // Update scroll dots
    updateScrollDots(scrollLeft, maxScroll);
}

function updateScrollDots(scrollLeft, maxScroll) {
    const indicator = document.getElementById('scrollIndicator');
    if (!indicator) return;

    const numDots = 5;
    const position = Math.floor((scrollLeft / maxScroll) * (numDots - 1));

    indicator.innerHTML = Array(numDots).fill(0).map((_, i) =>
        `<div class="scroll-dot ${i === position ? 'active' : ''}"></div>`
    ).join('');
}

function initCarouselScroll() {
    const container = document.getElementById('featuredModels');
    if (container) {
        container.addEventListener('scroll', () => {
            requestAnimationFrame(updateScrollIndicators);
        });

        // Initial state
        setTimeout(updateScrollIndicators, 500);
    }
}

// ============================================
// BACKTEST CAROUSEL
// ============================================

async function loadBacktestRankings() {
    try {
        const response = await fetch('/api/backtest-rankings');
        const data = await response.json();
        backtestData = data.rankings || [];
        renderBacktestModels();
    } catch (error) {
        console.error('Error loading backtest rankings:', error);
    }
}

function renderBacktestModels() {
    const container = document.getElementById('backtestModels');
    if (!container) return;

    if (backtestData.length === 0) {
        container.innerHTML = `<p style="color: var(--text-muted); padding: 20px;">No backtest data available yet</p>`;
        return;
    }

    container.innerHTML = backtestData.map(model => {
        const returnClass = model.cumulative_return >= 0 ? 'long' : 'short';
        const returnSign = model.cumulative_return > 0 ? '+' : '';

        return `
            <div class="featured-card" onclick="selectModel('${model.coin}', '${model.timeframe}')">
                <div class="featured-card-header">
                    <div class="featured-coin">
                        <img src="${model.logo_url}" alt="${model.coin}" class="coin-logo" onerror="this.style.display='none'">
                        ${model.coin}/${model.timeframe}
                    </div>
                    <span class="featured-prediction ${returnClass}">${returnSign}${model.cumulative_return.toFixed(0)}%</span>
                </div>
                <div class="featured-card-footer">
                    <span>Accuracy: ${model.accuracy}%</span>
                    <span>${model.total_predictions} trades</span>
                </div>
            </div>
        `;
    }).join('');

    initBacktestCarouselScroll();
}

function scrollBacktest(direction) {
    const container = document.getElementById('backtestModels');
    if (!container) return;

    const cardWidth = 300;
    const scrollAmount = cardWidth * 2;

    container.scrollBy({
        left: scrollAmount * direction,
        behavior: 'smooth'
    });

    setTimeout(() => updateBacktestScrollIndicators(), 300);
}

function updateBacktestScrollIndicators() {
    const container = document.getElementById('backtestModels');
    const wrapper = document.getElementById('backtestCarouselWrapper');
    const leftBtn = document.getElementById('scrollLeftBacktest');
    const rightBtn = document.getElementById('scrollRightBacktest');
    const indicator = document.getElementById('backtestScrollIndicator');

    if (!container || !wrapper) return;

    const scrollLeft = container.scrollLeft;
    const scrollWidth = container.scrollWidth;
    const clientWidth = container.clientWidth;
    const maxScroll = scrollWidth - clientWidth;

    // Update fade edges
    if (scrollLeft <= 10) {
        wrapper.className = 'featured-carousel-wrapper scrolled-start';
    } else if (scrollLeft >= maxScroll - 10) {
        wrapper.className = 'featured-carousel-wrapper scrolled-end';
    } else {
        wrapper.className = 'featured-carousel-wrapper scrolled-middle';
    }

    // Update arrow buttons
    if (leftBtn) leftBtn.disabled = scrollLeft <= 10;
    if (rightBtn) rightBtn.disabled = scrollLeft >= maxScroll - 10;

    // Update scroll dots
    if (indicator) {
        const numDots = 5;
        const position = maxScroll > 0 ? Math.floor((scrollLeft / maxScroll) * (numDots - 1)) : 0;
        indicator.innerHTML = Array(numDots).fill(0).map((_, i) =>
            `<div class="scroll-dot ${i === position ? 'active' : ''}"></div>`
        ).join('');
    }
}

function initBacktestCarouselScroll() {
    const container = document.getElementById('backtestModels');
    if (container) {
        container.addEventListener('scroll', () => {
            requestAnimationFrame(updateBacktestScrollIndicators);
        });
        setTimeout(updateBacktestScrollIndicators, 500);
    }
}

function renderFeaturedModels() {
    const featured = getFeaturedModels();

    if (featured.length === 0) {
        featuredModels.innerHTML = `<p style="color: var(--text-muted);">Loading top models...</p>`;
        return;
    }

    featuredModels.innerHTML = featured.map(model => {
        const isBest = model.accuracy >= 53;
        const signalClass = model.signal?.toLowerCase() || 'neutral';
        const predSign = model.prediction_pct > 0 ? '+' : '';

        return `
            <div class="featured-card ${isBest ? 'best' : ''}" onclick="selectModel('${model.coin}', '${model.timeframe}')">
                <div class="featured-card-header">
                    <div class="featured-coin">
                        <img src="${model.logo_url}" alt="${model.coin}" class="coin-logo" onerror="this.style.display='none'">
                        ${model.coin}/${model.timeframe}
                    </div>
                    <span class="featured-prediction ${signalClass}">${predSign}${model.prediction_pct.toFixed(2)}%</span>
                </div>
                <div class="featured-card-footer">
                    <div class="featured-accuracy">
                        ${isBest ? '<span class="badge badge-best">BEST</span>' : ''}
                        <span class="badge badge-pro">PRO</span>
                    </div>
                    <span>${formatPrice(model.current_price)}</span>
                </div>
            </div>
        `;
    }).join('');

    // Initialize carousel scroll after render
    initCarouselScroll();
}

function getBadges(accuracy) {
    let html = '<div class="badge-container">';
    if (accuracy >= 53) {
        html += '<span class="badge badge-best">BEST</span>';
        html += '<span class="badge badge-pro">PRO</span>';
    } else if (accuracy >= 52) {
        html += '<span class="badge badge-pro">PRO</span>';
    }
    html += '</div>';
    return html;
}

function renderModelList(modelData) {
    if (!modelData.length) {
        modelList.innerHTML = `<div class="loading-state"><p>No predictions available</p></div>`;
        return;
    }

    modelList.innerHTML = modelData.map((model) => {
        const isActive = selectedModel === model.model_key ? 'active' : '';
        const signalClass = model.signal?.toLowerCase() || 'neutral';
        const predSign = model.prediction_pct > 0 ? '+' : '';

        return `
            <div class="model-card ${isActive}" onclick="selectModel('${model.coin}', '${model.timeframe}')">
                <div class="model-card-header">
                    <span class="model-coin">
                        <img src="${model.logo_url}" alt="${model.coin}" class="coin-logo" onerror="this.style.display='none'">
                        ${model.coin}
                        <span class="model-tf">${model.timeframe}</span>
                    </span>
                    <span class="prediction-mini ${signalClass}">${predSign}${model.prediction_pct.toFixed(2)}%</span>
                </div>
                <div class="model-card-footer">
                    ${getBadges(model.accuracy)}
                    <span style="margin-left: auto;">${formatPrice(model.current_price)}</span>
                </div>
            </div>
        `;
    }).join('');
}

function renderModelDetail(analysisData, backtestData, chartData, modelInfo) {
    const analysis = analysisData.analysis || {};
    const pred = modelInfo || {};

    const signalClass = pred.signal?.toLowerCase() || 'neutral';
    const predColor = pred.prediction_pct > 0 ? '#22c55e' :
        pred.prediction_pct < 0 ? '#ef4444' : '#6b7280';

    // Backtest section
    let backtestHTML = '';
    if (backtestData && !backtestData.error) {
        const returnClass = backtestData.cumulative_return >= 0 ? 'positive' : 'negative';
        backtestHTML = `
            <div class="backtest-section">
                <h3>📊 Backtest (${backtestData.months_tested} Months)</h3>
                <div class="backtest-grid">
                    <div class="backtest-stat">
                        <div class="value">${backtestData.accuracy}%</div>
                        <div class="label">Accuracy</div>
                    </div>
                    <div class="backtest-stat">
                        <div class="value ${returnClass}">${backtestData.cumulative_return > 0 ? '+' : ''}${backtestData.cumulative_return.toFixed(2)}%</div>
                        <div class="label">Simulated P/L</div>
                    </div>
                    <div class="backtest-stat">
                        <div class="value">${backtestData.total_predictions}</div>
                        <div class="label">Trades</div>
                    </div>
                    <div class="backtest-stat">
                        <div class="value">${((backtestData.correct_direction / backtestData.total_predictions) * 100).toFixed(1)}%</div>
                        <div class="label">Win Rate</div>
                    </div>
                </div>
            </div>
        `;
    }

    // RSI color
    const rsiColor = pred.rsi > 70 ? '#ef4444' : pred.rsi < 30 ? '#22c55e' : '#6b6b7a';
    const rsiStatus = pred.rsi > 70 ? 'Overbought' : pred.rsi < 30 ? 'Oversold' : 'Neutral';

    detailPanel.innerHTML = `
        <div class="detail-header">
            <div class="detail-title">
                <h2>
                    <img src="${pred.logo_url || ''}" alt="${analysisData.coin}" class="coin-logo-large" onerror="this.style.display='none'">
                    ${analysisData.coin}/USDT
                    ${getBadges(pred.accuracy)}
                </h2>
                <span class="subtitle">${analysisData.timeframe} Timeframe</span>
            </div>
            <div class="prediction-box">
                <div class="prediction-label">AI Prediction</div>
                <div class="prediction-value" style="color: ${predColor}">
                    ${pred.prediction_pct > 0 ? '+' : ''}${pred.prediction_pct?.toFixed(4) || 'N/A'}%
                </div>
                <div class="prediction-signal ${signalClass}">${pred.signal || 'LOADING'}</div>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">${formatPrice(pred.current_price)}</div>
                <div class="label">Current Price</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: ${rsiColor}">${pred.rsi || 'N/A'}</div>
                <div class="label">RSI (${rsiStatus})</div>
            </div>
            <div class="stat-card">
                <div class="value">${analysis.baseline_accuracy?.toFixed(1) || pred.accuracy || 'N/A'}%</div>
                <div class="label">Model Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="value">${pred.signal === 'LONG' ? '📈' : pred.signal === 'SHORT' ? '📉' : '➡️'}</div>
                <div class="label">Expected Move</div>
            </div>
        </div>
        
        ${backtestHTML}
        
        <div class="chart-container">
            <h3>📈 Price Chart with Prediction</h3>
            <div class="chart-wrapper">
                <canvas id="priceChart"></canvas>
            </div>
        </div>
        
        <div class="disclaimer-note">
            ⚠️ AI predictions are for educational purposes only. Not financial advice. DYOR.
        </div>
    `;

    if (chartData && chartData.candles) {
        renderPriceChartWithPrediction(chartData, pred);
    }
}

function renderPriceChartWithPrediction(chartData, prediction) {
    const ctx = document.getElementById('priceChart')?.getContext('2d');
    if (!ctx) return;

    if (priceChart) priceChart.destroy();

    const candles = chartData.candles;
    const labels = candles.map((c, i) => {
        if (i === candles.length - 1) return 'NOW';
        const d = new Date(c.time);
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });

    labels.push('NEXT');

    const prices = candles.map(c => c.close);
    const lastPrice = prices[prices.length - 1];
    const predPct = prediction.prediction_pct || 0;
    const predictedPrice = lastPrice * (1 + predPct / 100);
    prices.push(predictedPrice);

    const isUp = predPct > 0;
    const lineColor = '#6366f1';
    const predColor = isUp ? '#22c55e' : '#ef4444';

    const pointColors = prices.map((_, i) => {
        if (i === prices.length - 1) return predColor;
        if (i === prices.length - 2) return '#ffffff';
        return 'transparent';
    });

    const pointRadius = prices.map((_, i) => {
        if (i === prices.length - 1) return 8;
        if (i === prices.length - 2) return 6;
        return 0;
    });

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price',
                data: prices,
                borderColor: lineColor,
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.2,
                pointBackgroundColor: pointColors,
                pointBorderColor: pointColors,
                pointRadius: pointRadius,
                borderWidth: 2,
                segment: {
                    borderColor: ctx => ctx.p1DataIndex === prices.length - 1 ? predColor : lineColor,
                    borderDash: ctx => ctx.p1DataIndex === prices.length - 1 ? [5, 5] : []
                }
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const idx = context.dataIndex;
                            if (idx === prices.length - 1) {
                                return `Predicted: ${formatPrice(context.parsed.y)} (${predPct > 0 ? '+' : ''}${predPct.toFixed(2)}%)`;
                            }
                            return formatPrice(context.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    ticks: { color: 'var(--text-muted)', maxTicksLimit: 10, maxRotation: 0 },
                    grid: { display: false }
                },
                y: {
                    ticks: {
                        color: 'var(--text-muted)',
                        callback: v => formatPrice(v)
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

// ============================================
// HELPERS
// ============================================

function selectModel(coin, timeframe) {
    selectedModel = `${coin}_${timeframe}`;

    document.querySelectorAll('.model-card, .featured-card').forEach(card => {
        card.classList.remove('active');
    });

    loadModelDetail(coin, timeframe);
}
