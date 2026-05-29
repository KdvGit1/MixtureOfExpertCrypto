/**
 * ⚡ Cyberpunk Overlay Controller — Battle Arena Coordinator
 * Integrates WebSockets, Web Audio API, Canvas Charts, Visual AI Simulation & UI States
 */

const state = {
    ethPrice: 0.0,
    ethHistory: [],
    soundEnabled: true,
    lastSignals: { "ETH/USDT": null },
    lang: localStorage.getItem("moe_lang") || "en",
    activeTimeframe: "15m",
    analysisData: null,
    
    // Trade Simulation Dashboard History lists
    closedTradesHistory: [],
    
    // Live Autonomous AI Bot Traders Simulation (Strictly locked to 5x leverage!)
    bots: {
        cnn: {
            name: "CNN Reflex Bot",
            role: "High-Frequency Scalper",
            balance: 10000.0,
            pnl: 0.0,
            position: "FLAT",
            entryPrice: 0.0,
            leverage: 5,
            tradingSymbol: "ETH/USDT",
            wins: 0,
            losses: 0
        },
        lstm: {
            name: "LSTM Historian Bot",
            role: "Trend Swinger",
            balance: 10000.0,
            pnl: 0.0,
            position: "FLAT",
            entryPrice: 0.0,
            leverage: 5,
            tradingSymbol: "ETH/USDT",
            wins: 0,
            losses: 0
        },
        tr: {
            name: "Transformer Macro Bot",
            role: "Macro Breakout",
            balance: 10000.0,
            pnl: 0.0,
            position: "FLAT",
            entryPrice: 0.0,
            leverage: 5,
            tradingSymbol: "ETH/USDT",
            wins: 0,
            losses: 0
        }
    }
};

// Web Audio API Synthesizer (Out-of-the-box Sound FX!)
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function playSynthSound(type) {
    if (!state.soundEnabled) return;
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }

    try {
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        const now = audioCtx.currentTime;

        if (type === "BUY") {
            // Ascending Cyber Chime
            osc.type = "sine";
            osc.frequency.setValueAtTime(587.33, now); // D5
            osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.15); // A5
            osc.frequency.exponentialRampToValueAtTime(1174.66, now + 0.3); // D6
            
            gainNode.gain.setValueAtTime(0.01, now);
            gainNode.gain.linearRampToValueAtTime(0.12, now + 0.05);
            gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
            
            osc.start(now);
            osc.stop(now + 0.5);
        } else if (type === "SELL") {
            // Descending Synth Buzz
            osc.type = "sawtooth";
            osc.frequency.setValueAtTime(329.63, now); // E4
            osc.frequency.exponentialRampToValueAtTime(220.00, now + 0.2); // A3
            osc.frequency.exponentialRampToValueAtTime(110.00, now + 0.45); // A2
            
            gainNode.gain.setValueAtTime(0.01, now);
            gainNode.gain.linearRampToValueAtTime(0.1, now + 0.1);
            gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
            
            osc.start(now);
            osc.stop(now + 0.55);
        } else if (type === "STREAK") {
            // Level Up Retro Chord
            osc.type = "triangle";
            osc.frequency.setValueAtTime(523.25, now); // C5
            osc.frequency.setValueAtTime(659.25, now + 0.08); // E5
            osc.frequency.setValueAtTime(783.99, now + 0.16); // G5
            osc.frequency.setValueAtTime(1046.50, now + 0.24); // C6
            
            gainNode.gain.setValueAtTime(0.01, now);
            gainNode.gain.linearRampToValueAtTime(0.18, now + 0.05);
            gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.55);
            
            osc.start(now);
            osc.stop(now + 0.6);
        } else if (type === "CLICK") {
            // High-pitched fast retro coin ding
            osc.type = "sine";
            osc.frequency.setValueAtTime(880, now);
            osc.frequency.exponentialRampToValueAtTime(1320, now + 0.04);
            
            gainNode.gain.setValueAtTime(0.01, now);
            gainNode.gain.linearRampToValueAtTime(0.06, now + 0.01);
            gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.06);
            
            osc.start(now);
            osc.stop(now + 0.07);
        }
    } catch (err) {
        console.error("Synth play failure: ", err);
    }
}

// WebSocket Connection Manager
function connectWebSocket() {
    const statusText = document.getElementById("statusText");
    const indicator = document.querySelector(".status-indicator");
    
    loggerText(locales[state.lang].websocket_connecting);
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "127.0.0.1:8888";
    const ws = new WebSocket(`${protocol}//${host}/ws/live`);

    ws.onopen = () => {
        loggerText(locales[state.lang].websocket_synced);
        statusText.innerText = locales[state.lang].websocket_online;
        indicator.className = "status-indicator online";
    };

    ws.onclose = () => {
        loggerText(locales[state.lang].websocket_lost);
        statusText.innerText = locales[state.lang].websocket_offline;
        indicator.className = "status-indicator offline";
        setTimeout(connectWebSocket, 4000);
    };

    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === "tick") {
            handleTickUpdate(payload.data);
        } else if (payload.type === "analysis") {
            handleAnalysisUpdate(payload.data);
        } else if (payload.type === "retrain_status") {
            handleRetrainStatusUpdate(payload.data);
        }
    };
}

function handleRetrainStatusUpdate(data) {
    const feed = document.getElementById("ordersFeed");
    if (!feed) return;

    const time = new Date().toLocaleTimeString(undefined, { hour12: false });
    const logItem = document.createElement("div");
    
    let itemClass = "long";
    if (data.action === "SYSTEM ERROR") {
        itemClass = "short";
    }
    
    logItem.className = `log-item ${itemClass}`;
    logItem.style.fontWeight = "bold";
    logItem.style.textShadow = "0 0 5px rgba(255, 183, 0, 0.4)";
    logItem.style.color = data.action === "SYSTEM SUCCESS" ? "#00ff66" : (data.action === "SYSTEM ERROR" ? "#ff007f" : "#ffb700");
    
    logItem.innerText = `[${time}] ⚙️ SYSTEM: ${data.message}`;
    
    feed.appendChild(logItem);
    feed.scrollTop = feed.scrollHeight;

    playSynthSound(data.action === "SYSTEM SUCCESS" ? "BUY" : "CLICK");

    while (feed.children.length > 25) {
        feed.removeChild(feed.firstChild);
    }
}

// Translate marquee logger ticker
function loggerText(text) {
    const crawl = document.getElementById("marqueeCrawl");
    if (crawl) {
        const lang = state.lang;
        const sub = lang === "tr" ? "CNN REFLEKSLERİ: AKTİF • LSTM TARİHÇİSİ: AKTİF • TRANSFORMER REJİMİ: AKTİF" : "CNN REFLEXES: ACTIVE • LSTM HISTORIAN: ACTIVE • TRANSFORMER REGIME: ACTIVE";
        crawl.innerText = `🚀 ${text.toUpperCase()} • ${sub}`;
    }
}

// Format price as currency helper
function formatPrice(val, isBtc) {
    return "$" + val.toLocaleString(undefined, { 
        minimumFractionDigits: isBtc ? 0 : 2, 
        maximumFractionDigits: isBtc ? 0 : 2 
    });
}

// Overridden drawPriceChart: plots price AND glowing predicted target coordinates with Y-axis price labels
function drawPriceChart(canvasId, history, color, isBtc) {
    const c = document.getElementById(canvasId);
    if (!c) return;
    const ctx = c.getContext("2d");
    
    const rect = c.parentElement.getBoundingClientRect();
    c.width = rect.width;
    c.height = rect.height;

    const w = c.width;
    const h = c.height;
    ctx.clearRect(0, 0, w, h);

    if (history.length < 2) return;

    const prices = history.map(t => t.price);
    let min = Math.min(...prices);
    let max = Math.max(...prices);

    // 1. Calculate prediction target price and adjust chart min/max limits
    let predictedPrice = null;
    let pred_main = 0.0;
    let hasPred = false;
    const symbol = isBtc ? "BTC/USDT" : "ETH/USDT";
    
    if (state.analysisData && state.analysisData[symbol] && state.analysisData[symbol][state.activeTimeframe]) {
        const info = state.analysisData[symbol][state.activeTimeframe];
        if (info.brain && info.brain.has_ai) {
            pred_main = info.brain.main;
            const lastPrice = prices[prices.length - 1];
            predictedPrice = lastPrice * (1 + pred_main);
            min = Math.min(min, predictedPrice);
            max = Math.max(max, predictedPrice);
            hasPred = true;
        }
    }

    const range = max - min || 1.0;
    
    // Add padding to top and bottom to avoid rendering lines right on the canvas edges
    const padTop = 20;
    const padBottom = 20;
    const chartHeight = h - padTop - padBottom;
    
    // We reserve the right 75px for the Y-axis labels and target points
    const chartWidth = w - 75;

    // Helper to calculate Y coordinate for a given price
    const getY = (price) => {
        return h - padBottom - ((price - min) / range) * chartHeight;
    };

    // 2. Draw Y-axis grid lines and price labels
    ctx.font = "normal 8px Orbitron";
    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
    ctx.textAlign = "left";
    
    const gridLevels = [
        { price: max },
        { price: min + range * 0.5 },
        { price: min }
    ];
    
    gridLevels.forEach(lvl => {
        const yGrid = getY(lvl.price);
        
        // Draw dashed grid line
        ctx.beginPath();
        ctx.setLineDash([2, 4]);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.moveTo(0, yGrid);
        ctx.lineTo(w, yGrid);
        ctx.stroke();
        ctx.setLineDash([]); // reset
        
        // Write price label
        const priceStr = formatPrice(lvl.price, isBtc);
        ctx.fillText(priceStr, chartWidth + 5, yGrid + 3);
    });

    // 3. Draw price history line (ends exactly at chartWidth)
    ctx.beginPath();
    history.forEach((tick, i) => {
        const x = (chartWidth / (history.length - 1)) * i;
        const y = getY(tick.price);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    // Stroke outline
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = color;
    ctx.stroke();
    ctx.shadowBlur = 0; // reset shadow

    // Fill area under line
    ctx.lineTo(chartWidth, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, hexToRgbA(color, 0.15));
    grad.addColorStop(1, hexToRgbA(color, 0.0));
    ctx.fillStyle = grad;
    ctx.fill();

    // 4. Draw last price dot (positioned exactly at chartWidth)
    const lastPrice = prices[prices.length - 1];
    const lastX = chartWidth;
    const lastY = getY(lastPrice);
    
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.shadowBlur = 8;
    ctx.shadowColor = color;
    ctx.fill();
    ctx.shadowBlur = 0;

    // 5. Draw target prediction dot and connecting dashed lines
    if (hasPred && predictedPrice !== null) {
        const predX = w - 10;
        const predY = getY(predictedPrice);
        const isUp = pred_main >= 0;
        const targetColor = isUp ? "#00ff66" : "#ff007f";
        
        // Draw dotted forecast link line
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(predX, predY);
        ctx.strokeStyle = targetColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]); // reset

        // Draw forecast target dot
        ctx.beginPath();
        ctx.arc(predX, predY, 5, 0, Math.PI * 2);
        ctx.fillStyle = targetColor;
        ctx.shadowBlur = 10;
        ctx.shadowColor = targetColor;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Render target price and prediction percentage badge
        ctx.font = "bold 9px Orbitron";
        ctx.fillStyle = targetColor;
        ctx.textAlign = "right";
        const pctText = `${isUp ? "▲" : "▼"} ${(pred_main * 100).toFixed(2)}%`;
        ctx.fillText(pctText, predX - 10, predY - 3);
    }
}

// Color conversion helper
function hexToRgbA(hex, alpha) {
    let c;
    if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
        c = hex.substring(1).split('');
        if (c.length == 3) {
            c = [c[0], c[0], c[1], c[1], c[2], c[2]];
        }
        c = '0x' + c.join('');
        return 'rgba(' + [(c >> 16) & 255, (c >> 8) & 255, c & 255].join(',') + ',' + alpha + ')';
    }
    return 'rgba(255,255,255,' + alpha + ')';
}

// 5s Tick Price Updates
function handleTickUpdate(data) {
    const info = data["ETH/USDT"];
    if (!info) return;
    
    state.ethPrice = info.price;
    state.ethHistory = info.history;
    
    // Update price text
    const priceEl = document.getElementById("eth-price");
    if (priceEl) {
        priceEl.innerText = `$${info.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }
    
    // Update highs/lows
    const highEl = document.getElementById("eth-high");
    const lowEl = document.getElementById("eth-low");
    if (highEl) highEl.innerText = `$${info.high24h.toLocaleString()}`;
    if (lowEl) lowEl.innerText = `$${info.low24h.toLocaleString()}`;
    
    // Change badge formatting
    const changeEl = document.getElementById("eth-change");
    if (changeEl) {
        const chg = info.change24h;
        changeEl.innerText = `${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%`;
        changeEl.className = `coin-change-badge ${chg >= 0 ? "positive" : "negative"}`;
    }
    
    // Draw Canvas Chart for ETH
    drawPriceChart("ethChart", state.ethHistory, "#ff007f", false);
    
    // Dynamic PnL simulator updates
    updateSimulatedTradersPnL();
}

// 60s Full AI Analysis
function handleAnalysisUpdate(data) {
    state.analysisData = data;
    renderAnalysisFrame();
}

// Render active timeframe analysis frame
function renderAnalysisFrame() {
    if (!state.analysisData) return;

    const data = state.analysisData;
    const timeframe = state.activeTimeframe;

    const tfInfo = data["ETH/USDT"] ? data["ETH/USDT"][timeframe] : null;
    if (!tfInfo) return;
    
    // 1. Decision Signal HUD Updates
    const signalVal = document.getElementById("eth-signal");
    const signalDesc = document.getElementById("eth-signal-desc");

    if (signalVal) {
        signalVal.innerText = tfInfo.signal;
        signalVal.className = `signal-value ${tfInfo.signal}`;
    }

    if (signalDesc) {
        signalDesc.innerText = tfInfo.signal_text;
    }

    // Add visual sparks alarms on signal changes
    if (state.lastSignals["ETH/USDT"] !== tfInfo.signal) {
        if (tfInfo.signal !== "HOLD") {
            playSynthSound(tfInfo.signal);
            const card = document.getElementById("card-ETH");
            if (card) {
                const rect = card.getBoundingClientRect();
                window.triggerSignalBurst(tfInfo.signal, rect.left + rect.width/2, rect.top + rect.height/2);
            }
        }
        state.lastSignals["ETH/USDT"] = tfInfo.signal;
    }

    // 3. Actionable Setup Levels computation & rendering
    const curPrice = state.ethPrice;
    if (curPrice > 0) {
        const pred_main = tfInfo.brain.main;
        let entry = curPrice;
        let tp1 = 0, tp2 = 0, sl = 0;

        if (tfInfo.signal === "BUY") {
            tp1 = curPrice * (1 + Math.abs(pred_main) * 0.5);
            tp2 = curPrice * (1 + Math.abs(pred_main));
            sl = curPrice * (1 - Math.abs(pred_main) * 0.45);
        } else if (tfInfo.signal === "SELL") {
            tp1 = curPrice * (1 - Math.abs(pred_main) * 0.5);
            tp2 = curPrice * (1 - Math.abs(pred_main));
            sl = curPrice * (1 + Math.abs(pred_main) * 0.45);
        } else {
            // HOLD default setup pivot boundaries
            tp1 = curPrice * 1.005;
            tp2 = curPrice * 1.012;
            sl = curPrice * 0.995;
        }

        document.getElementById("eth-entry-val").innerText = `$${entry.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById("eth-tp1-val").innerText = `$${tp1.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById("eth-tp2-val").innerText = `$${tp2.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById("eth-sl-val").innerText = `$${sl.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
    }

    // 4. Gating weight analysis (Dominant Path selection)
    const pathEl = document.getElementById("eth-dominant-path");
    if (pathEl) {
        const brain = tfInfo.brain;
        const cnnAbs = Math.abs(brain.cnn);
        const lstmAbs = Math.abs(brain.lstm);
        const trAbs = Math.abs(brain.tr);
        
        let dominant = "CNN Scalper";
        if (lstmAbs > cnnAbs && lstmAbs > trAbs) {
            dominant = "LSTM Swinger";
        } else if (trAbs > cnnAbs && trAbs > lstmAbs) {
            dominant = "Transformer Macro";
        }
        pathEl.innerText = dominant;
    }

    // 5. Tray bottom indicators updates
    const ind = tfInfo.indicators;
    const rsiEl = document.getElementById("eth-rsi");
    if (rsiEl) rsiEl.innerText = ind.rsi.toFixed(1);
    const bbEl = document.getElementById("eth-bb");
    if (bbEl) bbEl.innerText = ind.bb_pctb.toFixed(2);
    const volEl = document.getElementById("eth-vol");
    if (volEl) volEl.innerText = `${ind.vol_ratio.toFixed(1)}x`;

    // 6. Achievement overlay pops
    const ach = tfInfo.achievements;
    const rewardEl = document.getElementById("eth-achievement");
    if (rewardEl) {
        rewardEl.innerText = ach.event ? ach.event : "";
    }
    if (ach.event) {
        triggerAchievementAlert(ach.event);
    }

    // Run active AI Bot Trader Decisions
    simulateAIBotTrades();
}

// ==========================================================================
// 🤖 AUTONOMOUS AI BOTS SIMULATOR TRADING LOGIC
// ==========================================================================

// Render Open Positions, History, and Total Portfolio balance change% on the screen
function updateAutonomousTradeHUD() {
    const isTr = state.lang === "tr";
    
    // 1. Calculate unified total portfolio balance and PnL change
    let combinedVal = 0.0;
    const openPos = [];
    
    for (const [key, bot] of Object.entries(state.bots)) {
        let liveVal = bot.balance;
        if (bot.position !== "FLAT") {
            liveVal = bot.balance * (1 + bot.pnl / 100);
            
            // Calculate liquidation price
            const entryPrice = bot.entryPrice;
            const liqPrice = bot.position === "LONG" ? (entryPrice * 0.8) : (entryPrice * 1.2);
            
            openPos.push({
                botKey: key,
                botName: bot.name,
                direction: bot.position,
                asset: bot.tradingSymbol.split('/')[0],
                entryPrice: entryPrice,
                liqPrice: liqPrice,
                pnl: bot.pnl
            });
        }
        combinedVal += liveVal;
    }
    
    const initialCapital = 30000.0;
    const portfolioPnL = ((combinedVal - initialCapital) / initialCapital) * 100;
    
    const simBalEl = document.getElementById("sim-total-bal");
    const simPnLEl = document.getElementById("sim-total-pnl-pct");
    const simBoxEl = document.getElementById("sim-total-pnl-box");
    
    if (simBalEl) {
        simBalEl.innerText = `$${combinedVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    }
    if (simPnLEl) {
        simPnLEl.innerText = `${portfolioPnL >= 0 ? "+" : ""}${portfolioPnL.toFixed(2)}%`;
    }
    if (simBoxEl) {
        simBoxEl.className = `total-pnl-pct ${portfolioPnL >= 0 ? "profit" : "loss"}`;
    }

    // 2. Render Open Positions List
    const posList = document.getElementById("activePositionsList");
    if (posList) {
        posList.innerHTML = "";
        
        if (openPos.length === 0) {
            const noPos = document.createElement("div");
            noPos.className = "no-positions";
            noPos.id = "noPositionsMsg";
            noPos.innerText = isTr ? "Aktif açık işlem bulunmuyor." : "No active open positions.";
            posList.appendChild(noPos);
        } else {
            openPos.forEach(pos => {
                const row = document.createElement("div");
                row.className = "pos-row";
                
                const isBtc = pos.asset === "BTC";
                const entryFmt = formatPrice(pos.entryPrice, isBtc);
                const liqFmt = formatPrice(pos.liqPrice, isBtc);
                
                row.innerHTML = `
                    <div class="pos-bot-meta">
                        <span class="pos-direction-badge ${pos.direction}">${pos.direction}</span>
                        <span class="pos-bot-name">${pos.botName} (${pos.asset})</span>
                    </div>
                    <div class="pos-prices">
                        <span class="pos-price-entry">Entry: <strong>${entryFmt}</strong></span>
                        <span class="pos-price-liq">Liq: <strong>${liqFmt}</strong></span>
                    </div>
                    <div class="pos-pnl ${pos.pnl >= 0 ? "profit" : "loss"}">${pos.pnl >= 0 ? "+" : ""}${pos.pnl.toFixed(2)}%</div>
                `;
                posList.appendChild(row);
            });
        }
    }

    // 3. Render Completed Trades History Table
    const tableBody = document.getElementById("closedTradesTableBody");
    if (tableBody) {
        tableBody.innerHTML = "";
        
        state.closedTradesHistory.forEach(trade => {
            const tr = document.createElement("tr");
            
            let dirClass = trade.direction;
            let dirText = trade.direction;
            let pnlClass = trade.pnl >= 0 ? "profit" : "loss";
            let pnlText = `${trade.pnl >= 0 ? "+" : ""}${trade.pnl.toFixed(2)}%`;
            
            if (trade.direction === "LIQ") {
                dirClass = "LIQ";
                dirText = isTr ? "LİKİT 💀" : "LIQ 💀";
                pnlClass = "liquidated";
                pnlText = "-100.00%";
            }
            
            tr.innerHTML = `
                <td class="cell-bot">${trade.botName}</td>
                <td class="cell-asset">${trade.symbol}</td>
                <td class="cell-dir ${dirClass}">${dirText}</td>
                <td class="cell-pnl ${pnlClass}">${pnlText}</td>
            `;
            tableBody.appendChild(tr);
        });
        
        // Pad the table if less than 5 items to keep visuals beautiful
        if (state.closedTradesHistory.length === 0) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="4" style="text-align: center; color: var(--color-text-muted); font-style: italic; padding: 15px 0;">${isTr ? "Henüz tamamlanmış işlem yok." : "No completed trades yet."}</td>`;
            tableBody.appendChild(tr);
        }
    }
}

function simulateAIBotTrades() {
    if (!state.analysisData) return;

    const data = state.analysisData;
    const timeframe = state.activeTimeframe;

    // CNN Scalper Bot (Trades ETH on highly sensitive CNN signal)
    const cnnData = data["ETH/USDT"] ? data["ETH/USDT"][timeframe] : null;
    if (cnnData && cnnData.brain) {
        const cnnVal = cnnData.brain.cnn;
        const curPrice = state.ethPrice;
        let cnnSig = "FLAT";
        if (cnnVal > 0.0008) cnnSig = "LONG";
        else if (cnnVal < -0.0008) cnnSig = "SHORT";
        
        evaluateBotPosition("cnn", cnnSig, curPrice, "ETH/USDT");
    }

    // LSTM Historian Bot (Trades ETH on Trend signals)
    const lstmData = data["ETH/USDT"] ? data["ETH/USDT"][timeframe] : null;
    if (lstmData && lstmData.brain) {
        const lstmVal = lstmData.brain.lstm;
        const curPrice = state.ethPrice;
        let lstmSig = "FLAT";
        if (lstmVal > 0.0015) lstmSig = "LONG";
        else if (lstmVal < -0.0015) lstmSig = "SHORT";
        
        evaluateBotPosition("lstm", lstmSig, curPrice, "ETH/USDT");
    }

    // Transformer Macro Bot (Trades ETH on Macro break, no gate filtering blocks!)
    const trData = data["ETH/USDT"] ? data["ETH/USDT"][timeframe] : null;
    if (trData && trData.brain) {
        const trVal = trData.brain.tr;
        const curPrice = state.ethPrice;
        let trSig = "FLAT";
        
        if (trVal > 0.002) trSig = "LONG";
        else if (trVal < -0.002) trSig = "SHORT";
        
        evaluateBotPosition("tr", trSig, curPrice, "ETH/USDT");
    }
}

function evaluateBotPosition(botKey, targetPos, curPrice, symbol) {
    const bot = state.bots[botKey];
    if (bot.position === targetPos) return; // position unchanged

    // Complete active trade PnL calculations before switching
    if (bot.position !== "FLAT") {
        let tradePnL = 0.0;
        if (bot.position === "LONG") {
            tradePnL = ((curPrice - bot.entryPrice) / bot.entryPrice) * bot.leverage * 100;
        } else if (bot.position === "SHORT") {
            tradePnL = ((bot.entryPrice - curPrice) / bot.entryPrice) * bot.leverage * 100;
        }
        
        // Add final PnL to capital
        bot.balance *= (1 + tradePnL / 100);
        if (tradePnL >= 0) bot.wins++;
        else bot.losses++;
        
        logSimulationTrade(bot.name, "CLOSE", symbol, curPrice, tradePnL);
        
        // Log to Completed Trades History
        const tradeLog = {
            botName: bot.name,
            symbol: symbol.split('/')[0],
            direction: bot.position,
            entryPrice: bot.entryPrice,
            exitPrice: curPrice,
            pnl: tradePnL
        };
        state.closedTradesHistory.unshift(tradeLog);
        if (state.closedTradesHistory.length > 5) state.closedTradesHistory.pop();
    }

    // Enter new target position
    bot.position = targetPos;
    if (targetPos !== "FLAT") {
        bot.entryPrice = curPrice;
        logSimulationTrade(bot.name, targetPos, symbol, curPrice, 0);
        playSynthSound(targetPos === "LONG" ? "BUY" : "SELL");
        
        // Trigger spark alerts on bot card coordinates
        const el = document.getElementById(`bot-${botKey}`);
        if (el) {
            const rect = el.getBoundingClientRect();
            window.triggerSignalBurst(targetPos === "LONG" ? "BUY" : "SELL", rect.left + rect.width / 2, rect.top + rect.height / 2);
        }
    } else {
        bot.entryPrice = 0.0;
    }

    // Refresh bot card display and stats HUD
    updateBotDOMCard(botKey);
    updateAutonomousTradeHUD();
}

function updateSimulatedTradersPnL() {
    for (const [key, bot] of Object.entries(state.bots)) {
        if (bot.position === "FLAT") {
            bot.pnl = 0.0;
            continue;
        }

        const curPrice = state.ethPrice;
        if (curPrice <= 0 || bot.entryPrice <= 0) continue;

        // Calculate live PnL with 5x leverage
        if (bot.position === "LONG") {
            bot.pnl = ((curPrice - bot.entryPrice) / bot.entryPrice) * bot.leverage * 100;
        } else {
            bot.pnl = ((bot.entryPrice - curPrice) / bot.entryPrice) * bot.leverage * 100;
        }

        // --- CHECK LIQUIDATION EVENT ---
        // For 5x leverage, -20% price move from entry liquidates the position completely (-100% loss)
        const isLiquidated = (bot.position === "LONG" && curPrice <= bot.entryPrice * 0.8) || 
                            (bot.position === "SHORT" && curPrice >= bot.entryPrice * 1.2);
                            
        if (isLiquidated) {
            const entryPrice = bot.entryPrice;
            const liqPrice = bot.position === "LONG" ? (entryPrice * 0.8) : (entryPrice * 1.2);
            
            // Execute liquidation settlement
            bot.balance = 0.0; // fully liquidated to 0
            bot.position = "FLAT";
            bot.entryPrice = 0.0;
            bot.pnl = 0.0;
            bot.losses++;
            
            playSynthSound("SELL");
            
            // Log liquidation alert
            logSimulationTrade(bot.name, "LIQ", bot.tradingSymbol, liqPrice, -100.0);
            
            // Add liquidation to history table
            const tradeLog = {
                botName: bot.name,
                symbol: bot.tradingSymbol.split('/')[0],
                direction: "LIQ",
                entryPrice: entryPrice,
                exitPrice: liqPrice,
                pnl: -100.0
            };
            state.closedTradesHistory.unshift(tradeLog);
            if (state.closedTradesHistory.length > 5) state.closedTradesHistory.pop();
            
            // Trigger critical warning sparkle on DOM
            const el = document.getElementById(`bot-${key}`);
            if (el) {
                const rect = el.getBoundingClientRect();
                window.triggerSignalBurst("SELL", rect.left + rect.width / 2, rect.top + rect.height / 2);
            }
            
            // Funny Deposit margin refill mechanism to keep otonom trading running
            const botRef = bot;
            const keyRef = key;
            setTimeout(() => {
                botRef.balance = 10000.0; // deposited!
                logSimulationTrade(botRef.name, "REFILL", botRef.tradingSymbol, 0, 0);
                updateBotDOMCard(keyRef);
                updateAutonomousTradeHUD();
            }, 5000);
        }

        updateBotDOMCard(key);
    }
    
    // Refresh unified statistics display
    updateAutonomousTradeHUD();
}

function updateBotDOMCard(key) {
    const bot = state.bots[key];
    const card = document.getElementById(`bot-${key}`);
    if (!card) return;

    // Position badges update
    const specificBadge = card.querySelector(".bot-trade-badge");
    if (specificBadge) {
        specificBadge.innerText = bot.position === "FLAT" ? "FLAT" : (bot.position === "LONG" ? "LONG" : "SHORT");
        specificBadge.className = `bot-trade-badge ${bot.position}`;
    }

    // Live balance and dynamic PnL
    const pnlEl = card.querySelector(".stat-pnl");
    if (pnlEl) {
        pnlEl.innerText = `PnL: ${bot.pnl >= 0 ? "+" : ""}${bot.pnl.toFixed(2)}%`;
        pnlEl.className = `stat-pnl ${bot.pnl > 0 ? "profit" : (bot.pnl < 0 ? "loss" : "")}`;
    }

    const balEl = card.querySelector(".stat-bal");
    if (balEl) {
        const liveBal = bot.position === "FLAT" ? bot.balance : (bot.balance * (1 + bot.pnl / 100));
        balEl.innerText = `$${liveBal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    }

    // If active dynamic profit bounce avatar slightly
    const avatar = card.querySelector(".bot-avatar");
    if (avatar) {
        if (bot.pnl > 0.05) {
            avatar.style.transform = "scale(1.15) rotate(5deg)";
            avatar.style.borderColor = "var(--neon-green)";
        } else {
            avatar.style.transform = "none";
            avatar.style.borderColor = "rgba(255,255,255,0.1)";
        }
    }
}

function logSimulationTrade(botName, action, symbol, price, pnl) {
    const feed = document.getElementById("ordersFeed");
    if (!feed) return;

    const time = new Date().toLocaleTimeString(undefined, { hour12: false });
    const logItem = document.createElement("div");
    
    let actionStr = "";
    let itemClass = "flat";
    
    const isTr = state.lang === "tr";

    if (action === "LONG") {
        actionStr = isTr ? `5x Kaldıraçlı ALIM (LONG) girdi` : `entered 5x leverage LONG`;
        itemClass = "long";
    } else if (action === "SHORT") {
        actionStr = isStr ? `5x Kaldıraçlı SATIM (SHORT) girdi` : `entered 5x leverage SHORT`;
        itemClass = "short";
    } else if (action === "LIQ") {
        actionStr = isTr ? `⚠️ LİKİT OLDU! Bakiye sıfırlandı (-100%)` : `⚠️ WAS LIQUIDATED! Balance wiped out (-100%)`;
        itemClass = "short";
    } else if (action === "REFILL") {
        actionStr = isTr ? `🤖 Sistem Koruması: Ajan bakiyesi $10,000.00 ile yenilendi.` : `🤖 System Refill: Restored agent balance with $10,000.00.`;
        itemClass = "long";
    } else {
        const winLoss = pnl >= 0 ? (isTr ? "KÂR" : "PROFIT") : (isTr ? "ZARAR" : "LOSS");
        actionStr = isTr ? `işlemini kapattı: $${pnl.toFixed(2)}% ${winLoss}` : `closed position: ${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}% ${winLoss}`;
        itemClass = pnl >= 0 ? "long" : "short";
    }

    logItem.className = `log-item ${itemClass}`;
    if (action === "REFILL") {
        logItem.innerText = `[${time}] ${actionStr}`;
    } else {
        logItem.innerText = `[${time}] ${botName} ${actionStr} ${symbol.split('/')[0]} @ $${price.toLocaleString(undefined, {minimumFractionDigits: 1})}`;
    }
    
    feed.appendChild(logItem);
    feed.scrollTop = feed.scrollHeight; // auto-scroll

    // Limit active feeds to prevent massive memory overhead
    while (feed.children.length > 25) {
        feed.removeChild(feed.firstChild);
    }
}

// ==========================================================================
// 🌐 DICTIONARIES & TRANSLATIONS ENGINE
// ==========================================================================

const locales = {
    en: {
        brand: "MOE CORE",
        brand_subtitle: "v3.5 LIVE SCENE",
        websocket_online: "WEBSOCKET ONLINE",
        websocket_offline: "SOCKET OFFLINE - RECONNECTING",
        websocket_connecting: "CONNECTING NETWORK LAYER...",
        websocket_synced: "NETWORK NODE SYNCED SUCCESSFULLY.",
        websocket_lost: "CONNECTION LOST. REBOOTING SOCKET LINK...",
        sound_toggle: "Toggle Sound FX",
        lang_toggle: "🇹🇷 TR",
        btc_label: "BITCOIN PRO",
        eth_label: "ETHEREUM PRO",
        decision_gate: "DECISION GATE",
        waiting: "WAITING...",
        sync_matrix: "Synchronizing analysis matrix",
        xgb_gate: "XGBoost Gate",
        approved: "APPROVED",
        blocked: "BLOCKED",
        syncing: "Syncing",
        
        // Setup HUD
        setup_title: "📊 LIVE TRADING SETUP SCANNER",
        entry_lbl: "Recommended Entry",
        tp1_lbl: "Take Profit 1 (TP1)",
        tp2_lbl: "Take Profit 2 (TP2)",
        sl_lbl: "Stop Loss (SL)",
        dominant_path: "DOMINANT BRANCH:",
        
        // Simulation Arena
        arena_title: "🤖 NEURAL AUTONOMOUS SIMULATOR",
        arena_desc: "Live MoE Agent Trading Simulation",
        feed_title: "📜 LIVE SIMULATION CHRONICLE",
        open_pos_title: "⚡ ACTIVE OPEN POSITIONS",
        closed_trades_title: "📊 LAST 5 TRADES",
        
        achievement_title: "ACHIEVEMENT UNLOCKED",
        marquee: "🚀 MIXTURE OF EXPERTS NEURAL ENGINE ONLINE • CNN COGNITIVE REFLEXES: ACTIVE • LSTM TREND HISTORIAN: ACTIVE • TRANSFORMER MACRO REGIME SCANNER: ACTIVE • SCANNING BINANCE PERPETUAL CONTRACTS..."
    },
    tr: {
        brand: "MOE ÇEKİRDEK",
        brand_subtitle: "v3.5 CANLI PANEL",
        websocket_online: "WEBSOCKET ÇEVRİMİÇİ",
        websocket_offline: "SOKET ÇEVRİMDIŞI - BAĞLANIYOR",
        websocket_connecting: "AĞ KATMANI BAĞLANTISI KURULUYOR...",
        websocket_synced: "AĞ DÜĞÜMÜ BAŞARIYLA EŞLEŞTİRİLDİ.",
        websocket_lost: "BAĞLANTI KESİLDİ. SOKET YENİDEN BAŞLATILIYOR...",
        sound_toggle: "Ses Efektlerini Kapat/Aç",
        lang_toggle: "🇺🇸 EN",
        btc_label: "BITCOIN PRO",
        eth_label: "ETHEREUM PRO",
        decision_gate: "KARAR KAPISI",
        waiting: "BEKLENİYOR...",
        sync_matrix: "Analiz matrisi senkronize ediliyor",
        xgb_gate: "XGBoost Kapısı",
        approved: "ONAYLANDI",
        blocked: "ENGELLENDİ",
        syncing: "Senkronize Ediliyor",
        
        // Setup HUD
        setup_title: "📊 CANLI ALIM-SATIM KURULUM TARAYICI",
        entry_lbl: "Tavsiye Edilen Giriş",
        tp1_lbl: "Kâr Al 1 (TP1)",
        tp2_lbl: "Kâr Al 2 (TP2)",
        sl_lbl: "Zarar Durdur (SL)",
        dominant_path: "BASKIN UZMAN KOLU:",
        
        // Simulation Arena
        arena_title: "🤖 NÖRAL OTONOM SİMÜLATÖR",
        arena_desc: "Canlı MoE Ajan Alım-Satım Simülasyonu",
        feed_title: "📜 CANLI SİMÜLASYON GÜNCESİ",
        open_pos_title: "⚡ AKTİF AÇIK İŞLEMLER",
        closed_trades_title: "📊 SON 5 İŞLEM",
        
        achievement_title: "BAŞARI AÇILDI",
        marquee: "🚀 UZMANLAR KARIŞIMI (MoE) NÖRAL MOTOR ÇEVRİMİÇİ • CNN BİLİŞSEL REFLEKSLER: AKTİF • LSTM TREND TARİHÇİSİ: AKTİF • TRANSFORMER MAKRO REJİM TARAYICI: AKTİF • BİNANCE VADELİ İŞLEMLER TARANIYOR..."
    }
};

function updateLanguageUI() {
    const lang = state.lang;
    const l = locales[lang];
    
    // Header
    const brandTitle = document.querySelector(".hud-brand .hud-title");
    if (brandTitle) brandTitle.innerText = l.brand;
    const brandSub = document.querySelector(".hud-brand .hud-subtitle");
    if (brandSub) brandSub.innerText = l.brand_subtitle;
    
    const langToggleBtn = document.getElementById("langToggle");
    if (langToggleBtn) langToggleBtn.innerText = l.lang_toggle;
    const soundToggleBtn = document.getElementById("soundToggle");
    if (soundToggleBtn) soundToggleBtn.setAttribute("title", l.sound_toggle);
    
    // Coins Labels
    const ethLabel = document.getElementById("eth-label");
    if (ethLabel) ethLabel.innerText = l.eth_label;
    
    // Decision Gate
    const ethDecLabel = document.getElementById("eth-dec-label");
    if (ethDecLabel) ethDecLabel.innerText = l.decision_gate;
    
    // Setups Section Header
    const ethSetupH = document.getElementById("eth-setup-title");
    if (ethSetupH) ethSetupH.innerText = l.setup_title;
    
    // Setup labels
    const ethEntryLbl = document.getElementById("eth-entry-lbl");
    if (ethEntryLbl) ethEntryLbl.innerText = l.entry_lbl;
    const ethTp1Lbl = document.getElementById("eth-tp1-lbl");
    if (ethTp1Lbl) ethTp1Lbl.innerText = l.tp1_lbl;
    const ethTp2Lbl = document.getElementById("eth-tp2-lbl");
    if (ethTp2Lbl) ethTp2Lbl.innerText = l.tp2_lbl;
    const ethSlLbl = document.getElementById("eth-sl-lbl");
    if (ethSlLbl) ethSlLbl.innerText = l.sl_lbl;
    
    const ethPathLbl = document.getElementById("eth-path-lbl");
    if (ethPathLbl) ethPathLbl.innerText = l.dominant_path;

    // Simulation Card
    const arenaTitle = document.getElementById("arena-title");
    if (arenaTitle) arenaTitle.innerText = l.arena_title;
    const arenaDesc = document.getElementById("arena-desc");
    if (arenaDesc) arenaDesc.innerText = l.arena_desc;
    const feedTitle = document.getElementById("feed-title");
    if (feedTitle) feedTitle.innerText = l.feed_title;
    
    const openPosTitle = document.getElementById("lbl-open-pos-title");
    if (openPosTitle) openPosTitle.innerText = l.open_pos_title;
    const closedTradesTitle = document.getElementById("lbl-closed-trades-title");
    if (closedTradesTitle) closedTradesTitle.innerText = l.closed_trades_title;
    
    // Refresh waiting text
    const ethSignal = document.getElementById("eth-signal");
    if (ethSignal && (ethSignal.innerText === "WAITING..." || ethSignal.innerText === "BEKLENİYOR...")) {
        ethSignal.innerText = l.waiting;
    }
    const ethSignalDesc = document.getElementById("eth-signal-desc");
    if (ethSignalDesc && (ethSignalDesc.innerText === "Synchronizing analysis matrix" || ethSignalDesc.innerText === "Analiz matrisi senkronize ediliyor")) {
        ethSignalDesc.innerText = l.sync_matrix;
    }

    // Bots descriptions
    const bot1Role = document.getElementById("bot1-role");
    if (bot1Role) bot1Role.innerText = lang === "tr" ? "Yüksek-Frekans Refleks AL/SAT" : "High-Frequency Scalping";
    const bot2Role = document.getElementById("bot2-role");
    if (bot2Role) bot2Role.innerText = lang === "tr" ? "Trend-Takipçi Swinger Ajan" : "Trend-Following Swing";
    const bot3Role = document.getElementById("bot3-role");
    if (bot3Role) bot3Role.innerText = lang === "tr" ? "Makro Rejim Kırılma Dedektörü" : "Macro regime breakout";

    // Marquee content
    const marqueeCrawl = document.getElementById("marqueeCrawl");
    if (marqueeCrawl && (marqueeCrawl.innerText.startsWith("🚀 MIXTURE") || marqueeCrawl.innerText.startsWith("🚀 UZMANLAR"))) {
        marqueeCrawl.innerText = l.marquee;
    }
}

// Achievement gamification overlay triggers
function triggerAchievementAlert(text) {
    const overlay = document.getElementById("achievementOverlay");
    const nameEl = document.getElementById("achievementName");
    
    if (overlay && nameEl) {
        nameEl.innerText = text.substring(2); // Strip out emojis from text
        overlay.className = "game-achievement-overlay active";
        
        playSynthSound("STREAK");
        window.triggerSignalBurst("STREAK", window.innerWidth / 2, 120);

        setTimeout(() => {
            overlay.className = "game-achievement-overlay";
        }, 5000);
    }
}

// Timeframe tab clicks binding
// Timeframe tab clicks binding
function bindInteractiveEvents() {
    // Timeframe tab clicks deactivated - static scalper active

    // Language Toggle
    const langBtn = document.getElementById("langToggle");
    if (langBtn) {
        langBtn.addEventListener("click", () => {
            state.lang = state.lang === "en" ? "tr" : "en";
            localStorage.setItem("moe_lang", state.lang);
            updateLanguageUI();
            updateAutonomousTradeHUD(); // update stats panels localized strings
            playSynthSound("CLICK");
        });
    }

    // Sound toggle controls
    document.getElementById("soundToggle").addEventListener("click", () => {
        state.soundEnabled = !state.soundEnabled;
        document.getElementById("soundToggle").innerText = state.soundEnabled ? "🔊" : "🔇";
        playSynthSound("CLICK");
    });

    // Cyberpunk Settings Modal Toggles
    const settingsBtn = document.getElementById("settingsToggle");
    const settingsModal = document.getElementById("settingsModal");
    const closeSettingsBtn = document.getElementById("closeSettingsBtn");

    if (window.PUBLIC_MODE && settingsBtn) {
        settingsBtn.style.display = "none";
    }

    if (settingsBtn && settingsModal) {
        settingsBtn.addEventListener("click", () => {
            settingsModal.classList.add("active");
            playSynthSound("CLICK");
        });
    }

    if (closeSettingsBtn && settingsModal) {
        closeSettingsBtn.addEventListener("click", () => {
            settingsModal.classList.remove("active");
            playSynthSound("CLICK");
        });
    }

    if (settingsModal) {
        settingsModal.addEventListener("click", (e) => {
            if (e.target === settingsModal) {
                settingsModal.classList.remove("active");
                playSynthSound("CLICK");
            }
        });
    }

    // Bind Yerel Model Loader buttons clicks
    const bindModelLoader = (tf) => {
        const btn = document.getElementById(`load-model-${tf}-btn`);
        const input = document.getElementById(`model-path-${tf}`);
        const statusMsg = document.getElementById("config-status-msg");
        
        if (btn && input && statusMsg) {
            btn.addEventListener("click", () => {
                const path = input.value.trim();
                if (!path) {
                    statusMsg.className = "config-status error";
                    statusMsg.innerText = state.lang === "tr" ? "Lütfen geçerli bir dosya yolu girin!" : "Please enter a valid file path!";
                    playSynthSound("SELL");
                    return;
                }
                
                statusMsg.className = "config-status success";
                statusMsg.innerText = state.lang === "tr" ? "Model yükleniyor, lütfen bekleyin..." : "Loading model weights, please wait...";
                playSynthSound("CLICK");
                
                fetch("/api/config/models", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ timeframe: tf, path: path })
                })
                .then(res => res.json())
                .then(res => {
                    if (res.success) {
                        statusMsg.className = "config-status success";
                        statusMsg.innerText = state.lang === "tr" ? `Model Başarıyla Yüklendi! (${tf})` : `Model Loaded Successfully! (${tf})`;
                        playSynthSound("BUY");
                        logSimulationTrade("SYSTEM", "LOAD", tf, 0, 0);
                        
                        // Custom system notification in Chronicle
                        const time = new Date().toLocaleTimeString(undefined, { hour12: false });
                        const feed = document.getElementById("ordersFeed");
                        if (feed) {
                            const logItem = document.createElement("div");
                            logItem.className = "log-item long";
                            logItem.innerText = `[${time}] ⚙️ SYSTEM loaded custom ${tf} weights: ${path}`;
                            feed.appendChild(logItem);
                            feed.scrollTop = feed.scrollHeight;
                        }
                    } else {
                        statusMsg.className = "config-status error";
                        statusMsg.innerText = state.lang === "tr" ? `Yükleme hatası: ${res.message}` : `Load error: ${res.message}`;
                        playSynthSound("SELL");
                    }
                })
                .catch(err => {
                    statusMsg.className = "config-status error";
                    statusMsg.innerText = `Network error: ${err.message}`;
                    playSynthSound("SELL");
                });
            });
        }
    };
    
    bindModelLoader("15m");
}

// Boot application
updateLanguageUI();
updateAutonomousTradeHUD();
bindInteractiveEvents();
connectWebSocket();
