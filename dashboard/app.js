/**
 * NEUTRAL Grid Bot Validator - Frontend Application
 */

class GridValidator {
    constructor() {
        this.apiBase = '/api';
        this.isLoading = false;
        this.init();
    }

    init() {
        // Get DOM elements
        this.symbolInput = document.getElementById('symbol-input');
        this.validateBtn = document.getElementById('validate-btn');
        this.resultSection = document.getElementById('result-section');
        this.statusCard = document.getElementById('status-card');
        this.paramsCard = document.getElementById('params-card');
        this.detailsCard = document.getElementById('details-card');

        // Bind events
        this.validateBtn.addEventListener('click', () => this.validate());
        this.symbolInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.validate();
        });

        // Quick pair buttons
        document.querySelectorAll('.quick-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.symbolInput.value = btn.dataset.symbol;
                this.validate();
            });
        });
    }

    async validate() {
        const symbol = this.symbolInput.value.trim().toUpperCase();

        if (!symbol) {
            this.showError('Please enter a trading pair');
            return;
        }

        if (this.isLoading) return;

        this.setLoading(true);

        try {
            const response = await fetch(`${this.apiBase}/validate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symbol }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.displayResult(data);
        } catch (error) {
            console.error('Validation error:', error);
            this.showError(`Error: ${error.message}`);
        } finally {
            this.setLoading(false);
        }
    }

    setLoading(loading) {
        this.isLoading = loading;
        this.validateBtn.classList.toggle('loading', loading);
        this.validateBtn.disabled = loading;
    }

    displayResult(data) {
        // Show result section
        this.resultSection.style.display = 'block';

        // Update status card
        document.getElementById('status-symbol').textContent = data.symbol;
        const statusBadge = document.getElementById('status-badge');
        statusBadge.textContent = data.is_valid ? 'VALID' : 'INVALID';
        statusBadge.className = `status-badge ${data.is_valid ? 'valid' : 'invalid'}`;

        const statusMessage = document.getElementById('status-message');
        if (data.is_valid) {
            statusMessage.textContent = 'This pair passes all multi-timeframe regime validation checks.';
        } else {
            statusMessage.textContent = 'This pair does not meet NEUTRAL grid requirements.';
        }

        // Show/hide params card
        if (data.is_valid && data.grid_params) {
            this.displayParams(data.grid_params);
            this.paramsCard.style.display = 'block';
        } else {
            this.paramsCard.style.display = 'none';
        }

        // Show validation details
        if (data.validation_details) {
            this.displayDetails(data.validation_details);
            this.detailsCard.style.display = 'block';
        } else {
            this.detailsCard.style.display = 'none';
        }

        // Scroll to results
        this.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    displayParams(params) {
        // Grid range
        if (params.grid_lower && params.grid_upper) {
            document.getElementById('param-range').textContent =
                `$${this.formatPrice(params.grid_lower)} - $${this.formatPrice(params.grid_upper)}`;
        }

        // Number of grids
        document.getElementById('param-grids').textContent = params.num_grids || '-';

        // Grid spacing
        document.getElementById('param-spacing').textContent =
            params.grid_spacing_pct ? `${params.grid_spacing_pct.toFixed(3)}%` : '-';

        // Profit per grid
        document.getElementById('param-profit').textContent =
            params.profit_per_grid_pct ? `${params.profit_per_grid_pct.toFixed(4)}%` : '-';

        // Leverage
        document.getElementById('param-leverage').textContent =
            params.leverage ? `${params.leverage}x` : '-';

        // Total notional
        document.getElementById('param-notional').textContent =
            params.total_notional ? `$${params.total_notional.toFixed(2)}` : '-';

        // Expected return
        document.getElementById('param-return').textContent =
            params.expected_net_return_pct ? `+${params.expected_net_return_pct.toFixed(2)}%` : '-';

        // SL/TP
        document.getElementById('param-sltp').textContent =
            `${params.stop_loss_pct}% / +${params.take_profit_pct}%`;

        // Max hold time
        document.getElementById('param-time').textContent = params.max_holding_time || '-';

        // Fees
        document.getElementById('param-fees').textContent =
            `${params.maker_fee_pct}% / ${params.taker_fee_pct}%`;
    }

    displayDetails(details) {
        const timeframes = ['1h', '15m', '5m'];

        timeframes.forEach(tf => {
            const checkEl = document.getElementById(`check-${tf}`);
            const detailsEl = document.getElementById(`details-${tf}`);
            const data = details[tf];

            if (!data) {
                checkEl.className = 'tf-check';
                checkEl.querySelector('.tf-status').textContent = 'N/A';
                detailsEl.innerHTML = '';
                return;
            }

            // Set status
            checkEl.className = `tf-check ${data.passed ? 'passed' : 'failed'}`;
            checkEl.querySelector('.tf-status').textContent = data.passed ? 'PASS' : 'FAIL';

            // Build details
            let detailsHtml = '';

            if (data.checks) {
                const checks = data.checks;

                if (tf === '1h') {
                    detailsHtml += this.makeDetail('ADX', checks.adx, checks.adx_valid);
                    detailsHtml += this.makeDetail('EMA Flat', checks.ema_slopes_flat);
                    detailsHtml += this.makeDetail('Converging', checks.emas_converging);
                    detailsHtml += this.makeDetail('Trend', checks.trend_structure, checks.trend_valid);
                    detailsHtml += this.makeDetail('BB Contract', checks.bb_contracting);
                }

                if (tf === '15m') {
                    detailsHtml += this.makeDetail('ADX', checks.adx, checks.adx_valid);
                    detailsHtml += this.makeDetail('RSI', checks.rsi_mean, checks.rsi_oscillating);
                    detailsHtml += this.makeDetail('In Inner 60%', checks.price_in_inner);
                    detailsHtml += this.makeDetail('EMA Flat', checks.ema_flat);
                    detailsHtml += this.makeDetail('Range', `${checks.range_size_pct}%`);
                }

                if (tf === '5m') {
                    detailsHtml += this.makeDetail('ADX', checks.adx, checks.adx_valid);
                    detailsHtml += this.makeDetail('EMA Crosses', checks.ema_crosses, checks.ema_crosses_valid);
                    detailsHtml += this.makeDetail('VWAP Crosses', checks.vwap_crosses, checks.vwap_crosses_valid);
                    detailsHtml += this.makeDetail('No Drift', checks.no_drift);
                    detailsHtml += this.makeDetail('Drift', `${checks.drift_pct}%`);
                }
            }

            if (data.reason && !data.passed) {
                detailsHtml += `<span class="tf-detail fail">⚠ ${data.reason}</span>`;
            }

            detailsEl.innerHTML = detailsHtml;
        });
    }

    makeDetail(label, value, isValid = null) {
        let displayValue = value;
        let className = '';

        if (typeof value === 'boolean') {
            displayValue = value ? '✓' : '✗';
            className = value ? 'pass' : 'fail';
        } else if (isValid !== null) {
            className = isValid ? 'pass' : 'fail';
            if (typeof value === 'number') {
                displayValue = value.toFixed ? value.toFixed(2) : value;
            }
        }

        return `<span class="tf-detail ${className}">${label}: ${displayValue}</span>`;
    }

    formatPrice(price) {
        if (price >= 1000) {
            return price.toFixed(2);
        } else if (price >= 1) {
            return price.toFixed(4);
        } else {
            return price.toFixed(6);
        }
    }

    showError(message) {
        // Show result section with error
        this.resultSection.style.display = 'block';

        document.getElementById('status-symbol').textContent = 'ERROR';
        const statusBadge = document.getElementById('status-badge');
        statusBadge.textContent = 'ERROR';
        statusBadge.className = 'status-badge invalid';
        document.getElementById('status-message').textContent = message;

        this.paramsCard.style.display = 'none';
        this.detailsCard.style.display = 'none';
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    window.validator = new GridValidator();
});
