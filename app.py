import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy import stats
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Ricardo's Trading Strategy - Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize session state for settings ---
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'cointegration_window': 252,  # 1 year for daily data
        'cointegration_pvalue_threshold': 0.05,
        'correlation_window': 120,
        'correlation_threshold': 0.5,
        'beta_window': 30,
        'beta_min': 1.0,
        'beta_max': 2.0,
        'mad_window': 120,
        'mad_k_entry': 2.0,  # MAD multiplier for entry
        'mad_k_exit': 0.0,   # MAD multiplier for exit (median cross)
        'use_ewma_beta': False,
        'ewma_decay_factor': 0.94,
        'strong_coin': 'ETH',
        'weak_coin': 'BCH',
        'base_capital': 10000,
        'max_ltv': 0.75
    }

# --- Custom CSS for Professional Purple Theme ---
st.markdown("""
<style>
    :root {
        --fractal-primary: #6366f1;
        --fractal-secondary: #4f46e5;
        --fractal-accent: #8b5cf6;
        --fractal-purple: #7c3aed;
        --fractal-purple-light: #a78bfa;
        --fractal-purple-dark: #5b21b6;
        --fractal-success: #10b981;
        --fractal-warning: #f59e0b;
        --fractal-danger: #ef4444;
        --fractal-dark: #1e1b4b;
        --fractal-light: #f8fafc;
        --fractal-gray: #64748b;
        --fractal-gray-light: #e2e8f0;
        --fractal-white: #ffffff;
    }
    
    /* Global styling with purple theme */
    .main {
        background: linear-gradient(135deg, #faf5ff 0%, #ede9fe 50%, #ddd6fe 100%);
        padding: 24px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        min-height: 100vh;
    }
    
    /* Enhanced metric cards with purple accents */
    .metric-card {
        background: linear-gradient(135deg, var(--fractal-white) 0%, #fefbff 100%);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.15), 0 4px 16px rgba(99, 102, 241, 0.1);
        margin-bottom: 28px;
        border: 1px solid rgba(139, 92, 246, 0.2);
        backdrop-filter: blur(16px);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, var(--fractal-primary), var(--fractal-accent), var(--fractal-purple));
        border-radius: 20px 20px 0 0;
    }
    
    /* Stats table styling */
    .stats-table {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .stats-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .stats-table th {
        background: linear-gradient(135deg, var(--fractal-primary), var(--fractal-accent));
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    
    .stats-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(139, 92, 246, 0.1);
    }
    
    .stats-table tr:hover {
        background: rgba(139, 92, 246, 0.05);
    }
    
    /* Chart containers */
    .chart-container {
        background: linear-gradient(135deg, var(--fractal-white) 0%, #fefbff 100%);
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 12px 48px rgba(139, 92, 246, 0.12), 0 4px 20px rgba(99, 102, 241, 0.08);
        margin-bottom: 28px;
        border: 1px solid rgba(139, 92, 246, 0.15);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, var(--fractal-primary), var(--fractal-accent)) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, var(--fractal-white), #fefbff);
        border-radius: 16px;
        padding: 12px;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: var(--fractal-purple);
        font-weight: 700;
        padding: 16px 24px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--fractal-primary), var(--fractal-accent)) !important;
        color: var(--fractal-white) !important;
        box-shadow: 0 4px 16px rgba(139, 92, 246, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid var(--fractal-primary);
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid var(--fractal-warning);
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid var(--fractal-success);
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the trading data"""
    try:
        df = pd.read_excel('ETH_BCH_multi_interval_analysis.xlsx')
        df['ISO Date'] = pd.to_datetime(df['ISO Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_cointegration(price1, price2, window=None):
    """Calculate Engle-Granger cointegration test"""
    if window is not None and len(price1) > window:
        price1 = price1.tail(window)
        price2 = price2.tail(window)
    
    try:
        # Engle-Granger test
        score, pvalue, _ = coint(price1, price2)
        return {
            'score': score,
            'pvalue': pvalue,
            'is_cointegrated': pvalue < 0.05,
            'confidence': 'Strong' if pvalue < 0.01 else 'Moderate' if pvalue < 0.05 else 'Weak'
        }
    except:
        return {
            'score': np.nan,
            'pvalue': 1.0,
            'is_cointegrated': False,
            'confidence': 'Failed'
        }

def calculate_rolling_cointegration(price1, price2, window=252):
    """Calculate rolling cointegration p-values"""
    pvalues = []
    dates = []
    
    for i in range(window, len(price1)):
        window_price1 = price1.iloc[i-window:i]
        window_price2 = price2.iloc[i-window:i]
        
        result = calculate_cointegration(window_price1, window_price2)
        pvalues.append(result['pvalue'])
        # Use the actual date from the dataframe instead of price series index
        dates.append(price1.index[i])
    
    return pd.Series(pvalues, index=dates)

def calculate_ewma_beta(returns1, returns2, decay_factor=0.94):
    """Calculate EWMA (Exponentially Weighted Moving Average) Beta"""
    # Calculate EWMA variances and covariance
    ewma_var1 = returns1.ewm(alpha=1-decay_factor).var()
    ewma_cov = returns1.ewm(alpha=1-decay_factor).cov(returns2)
    
    # Beta = Cov(weak, strong) / Var(strong)
    ewma_beta = ewma_cov / ewma_var1
    return ewma_beta

def calculate_mad_z_scores(series, window=120):
    """Calculate MAD-based Z-scores for robust statistical analysis"""
    mad_z_scores = []
    medians = []
    mads = []
    
    for i in range(len(series)):
        if i < window:
            mad_z_scores.append(np.nan)
            medians.append(np.nan)
            mads.append(np.nan)
            continue
            
        # Get rolling window
        window_data = series.iloc[i-window:i]
        
        # Calculate median
        median_val = window_data.median()
        medians.append(median_val)
        
        # Calculate MAD (Median Absolute Deviation)
        mad = np.median(np.abs(window_data - median_val))
        mads.append(mad)
        
        # Convert to sigma-like scale
        sigma_equivalent = 1.4826 * mad
        
        # Calculate Z-score
        if sigma_equivalent > 0:
            z_score = (series.iloc[i] - median_val) / sigma_equivalent
        else:
            z_score = 0
            
        mad_z_scores.append(z_score)
    
    return pd.Series(mad_z_scores, index=series.index), pd.Series(medians, index=series.index), pd.Series(mads, index=series.index)

def calculate_spread_statistics(spread_series, window=120):
    """Calculate comprehensive statistics for spread analysis"""
    stats = {}
    
    # Basic statistics
    stats['mean'] = spread_series.mean()
    stats['median'] = spread_series.median()
    stats['std'] = spread_series.std()
    stats['mad'] = np.median(np.abs(spread_series - spread_series.median()))
    stats['min'] = spread_series.min()
    stats['max'] = spread_series.max()
    
    # Percentiles
    stats['p05'] = spread_series.quantile(0.05)
    stats['p20'] = spread_series.quantile(0.20)
    stats['p50'] = spread_series.quantile(0.50)
    stats['p80'] = spread_series.quantile(0.80)
    stats['p95'] = spread_series.quantile(0.95)
    
    # MAD-based thresholds
    stats['mad_2_lower'] = stats['median'] - 2 * stats['mad'] * 1.4826
    stats['mad_2_upper'] = stats['median'] + 2 * stats['mad'] * 1.4826
    
    return stats

def calculate_half_life(spread_series):
    """Calculate half-life of mean reversion using OLS"""
    spread_lag = spread_series.shift(1)
    spread_diff = spread_series - spread_lag
    spread_lag = spread_lag[1:]
    spread_diff = spread_diff[1:]
    
    # OLS regression: y_t - y_{t-1} = alpha + beta * y_{t-1} + epsilon
    X = spread_lag.values.reshape(-1, 1)
    y = spread_diff.values
    
    # Add constant
    X = np.column_stack([np.ones(len(X)), X])
    
    try:
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        half_life = -np.log(2) / beta[1] if beta[1] < 0 else np.inf
        return max(1, min(half_life, 365))  # Cap between 1 and 365 days
    except:
        return 60  # Default half-life

def generate_trading_signals(df, settings):
    """Generate trading signals based on MAD approach"""
    # Calculate spread
    df['log_spread'] = np.log(df['ETH Close']) - np.log(df['BCH Close'])
    
    # Calculate MAD Z-scores
    mad_z_scores, medians, mads = calculate_mad_z_scores(df['log_spread'], window=settings['mad_window'])
    df['mad_z_score'] = mad_z_scores
    df['spread_median'] = medians
    df['spread_mad'] = mads
    
    # Generate signals
    df['entry_signal'] = (df['mad_z_score'] <= -settings['mad_k_entry']).astype(int)
    df['exit_signal'] = (df['mad_z_score'] >= settings['mad_k_exit']).astype(int)
    
    # Apply risk filters
    if settings['use_ewma_beta']:
        df['beta'] = calculate_ewma_beta(df['LN BCH Var %'], df['LN ETH Var %'], settings['ewma_decay_factor'])
    else:
        df['beta'] = df['Rolling Beta']
    
    # Risk conditions
    corr_ok = df['Rolling Corr'] >= settings['correlation_threshold']
    beta_ok = (df['beta'] >= settings['beta_min']) & (df['beta'] <= settings['beta_max'])
    risk_ok = corr_ok & beta_ok
    
    # Only allow signals when risk filters pass
    df['entry_signal'] = (df['entry_signal'] & risk_ok).astype(int)
    
    return df

def main():
    st.title("📊 Ricardo's Statistical Arbitrage Strategy")
    st.markdown("### Focus: Cointegration, MAD-based signals, and proper statistical analysis")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Strategy Parameters")
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['ISO Date'].min(), df['ISO Date'].max()),
        min_value=df['ISO Date'].min(),
        max_value=df['ISO Date'].max()
    )
    
    # Filter data by date range
    if len(date_range) == 2:
        mask = (df['ISO Date'] >= pd.Timestamp(date_range[0])) & (df['ISO Date'] <= pd.Timestamp(date_range[1]))
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
    
    # Statistical parameters
    st.sidebar.subheader("📐 Statistical Parameters")
    
    st.session_state.settings['cointegration_window'] = st.sidebar.slider(
        "Cointegration Window (days)", 60, 500, st.session_state.settings['cointegration_window'],
        help="Window for cointegration test. Daily data uses ~252 days (1 year)"
    )
    
    st.session_state.settings['mad_window'] = st.sidebar.slider(
        "MAD Window (days)", 30, 250, st.session_state.settings['mad_window'],
        help="Window for MAD calculation"
    )
    
    st.session_state.settings['mad_k_entry'] = st.sidebar.slider(
        "MAD Entry Threshold (k)", 1.0, 3.0, st.session_state.settings['mad_k_entry'], 0.1,
        help="Enter when spread is k MADs below median"
    )
    
    # Risk parameters
    st.sidebar.subheader("🛡️ Risk Parameters")
    
    st.session_state.settings['correlation_threshold'] = st.sidebar.slider(
        "Min Correlation", 0.0, 1.0, st.session_state.settings['correlation_threshold'], 0.05,
        help="Minimum correlation required"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state.settings['beta_min'] = st.number_input(
            "Min Beta", 0.5, 2.0, st.session_state.settings['beta_min'], 0.1
        )
    with col2:
        st.session_state.settings['beta_max'] = st.number_input(
            "Max Beta", 1.0, 3.0, st.session_state.settings['beta_max'], 0.1
        )
    
    st.session_state.settings['use_ewma_beta'] = st.sidebar.checkbox(
        "Use EWMA Beta", st.session_state.settings['use_ewma_beta'],
        help="Use exponentially weighted beta for faster reaction to volatility changes"
    )
    
    if st.session_state.settings['use_ewma_beta']:
        st.session_state.settings['ewma_decay_factor'] = st.sidebar.slider(
            "EWMA Decay Factor", 0.90, 0.99, st.session_state.settings['ewma_decay_factor'], 0.01
        )
    
    # Generate signals
    filtered_df = generate_trading_signals(filtered_df, st.session_state.settings)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 1. Cointegration Analysis",
        "📊 2. Statistical Metrics",
        "📈 3. MAD-Based Signals",
        "🎯 4. Trading Opportunities",
        "📋 5. Position Management"
    ])
    
    with tab1:
        st.header("🔍 Cointegration Analysis")
        st.markdown("""
        <div class="info-box">
        <b>Why Cointegration is CRUCIAL:</b><br>
        Cointegration tests whether the log price spread between ETH and BCH is mean-reverting over the long run. 
        This is the foundation of our mean-reversion strategy:<br>
        • <b>P-value < 0.05:</b> Statistical evidence that the spread will revert to its mean<br>
        • <b>P-value ≥ 0.05:</b> No evidence of mean reversion - strategy may fail<br>
        • <b>Rolling Analysis:</b> Shows when the relationship strengthens or weakens over time
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        # Current cointegration test
        current_coint = calculate_cointegration(
            np.log(filtered_df['ETH Close']), 
            np.log(filtered_df['BCH Close']),
            window=st.session_state.settings['cointegration_window']
        )
        
        with col1:
            color = "success" if current_coint['is_cointegrated'] else "warning"
            st.markdown(f"""
            <div class="{color}-box">
            <h4>🎯 Current Cointegration Status</h4>
            <p><b>P-value:</b> {current_coint['pvalue']:.4f}</p>
            <p><b>Status:</b> {"✅ COINTEGRATED" if current_coint['is_cointegrated'] else "⚠️ NOT COINTEGRATED"}</p>
            <p><b>Confidence:</b> {current_coint['confidence']}</p>
            <p><b>Strategy Viability:</b> {"🟢 GOOD" if current_coint['is_cointegrated'] else "🔴 RISKY"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Calculate half-life
            half_life = calculate_half_life(filtered_df['log_spread'])
            st.markdown(f"""
            <div class="info-box">
            <h4>⏱️ Mean Reversion Speed</h4>
            <p><b>Half-life:</b> {half_life:.1f} days</p>
            <p><b>Full Reversion:</b> ~{int(half_life * 3)} days</p>
            <p><b>Suggested Window:</b> {int(half_life * 2)} days</p>
            <p><b>Position Duration:</b> {int(half_life * 1.5)}-{int(half_life * 2)} days</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Latest correlation and beta
            latest_corr = filtered_df['Rolling Corr'].iloc[-1]
            latest_beta = filtered_df['beta'].iloc[-1]
            
            risk_ok = (latest_corr >= st.session_state.settings['correlation_threshold'] and 
                      st.session_state.settings['beta_min'] <= latest_beta <= st.session_state.settings['beta_max'])
            
            color = "success" if risk_ok else "warning"
            st.markdown(f"""
            <div class="{color}-box">
            <h4>🛡️ Risk Filter Status</h4>
            <p><b>Correlation:</b> {latest_corr:.3f} (≥{st.session_state.settings['correlation_threshold']:.2f})</p>
            <p><b>Beta:</b> {latest_beta:.3f} ({st.session_state.settings['beta_min']:.1f}-{st.session_state.settings['beta_max']:.1f})</p>
            <p><b>Status:</b> {"✅ PASS ALL FILTERS" if risk_ok else "⚠️ FAIL FILTERS"}</p>
            <p><b>Trade Signal:</b> {"🟢 ALLOWED" if risk_ok else "🔴 BLOCKED"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Rolling cointegration chart
        st.subheader("📈 Rolling Cointegration Analysis - The Foundation")
        
        st.markdown("""
        <div class="warning-box">
        <b>Critical:</b> This chart shows the strength of the mean-reversion relationship over time. 
        Green areas (p < 0.05) indicate periods where the strategy has statistical backing. 
        Red areas indicate periods where mean-reversion is not statistically significant.
        </div>
        """, unsafe_allow_html=True)
        
        if len(filtered_df) > st.session_state.settings['cointegration_window']:
            # Create price series with proper datetime index
            eth_log_prices = pd.Series(np.log(filtered_df['ETH Close'].values), 
                                     index=filtered_df['ISO Date'])
            bch_log_prices = pd.Series(np.log(filtered_df['BCH Close'].values), 
                                     index=filtered_df['ISO Date'])
            
            rolling_pvalues = calculate_rolling_cointegration(
                eth_log_prices, 
                bch_log_prices,
                window=st.session_state.settings['cointegration_window']
            )
            
            fig_coint = go.Figure()
            
            # Add p-value line
            fig_coint.add_trace(go.Scatter(
                x=rolling_pvalues.index,
                y=rolling_pvalues.values,
                name='Cointegration P-value',
                line=dict(color='#6366f1', width=3),
                hovertemplate='Date: %{x}<br>P-value: %{y:.4f}<br>Status: %{text}',
                text=['Cointegrated' if p < 0.05 else 'Not Cointegrated' for p in rolling_pvalues.values]
            ))
            
            # Add significance thresholds
            fig_coint.add_hline(y=0.05, line_dash="solid", line_color="red", line_width=2,
                               annotation_text="5% Significance Level (Critical)")
            fig_coint.add_hline(y=0.01, line_dash="dash", line_color="green", line_width=2,
                               annotation_text="1% Strong Significance")
            
            # Shade cointegrated regions (p < 0.05)
            cointegrated_periods = []
            start_date = None
            
            for i, (date, pval) in enumerate(rolling_pvalues.items()):
                if pval < 0.05 and start_date is None:
                    start_date = date
                elif pval >= 0.05 and start_date is not None:
                    cointegrated_periods.append((start_date, date))
                    start_date = None
            
            # Handle case where cointegration extends to the end
            if start_date is not None:
                cointegrated_periods.append((start_date, rolling_pvalues.index[-1]))
            
            # Add shaded regions
            for start, end in cointegrated_periods:
                fig_coint.add_vrect(
                    x0=start, x1=end,
                    fillcolor="green", opacity=0.2, line_width=0,
                    annotation_text="Cointegrated Period" if start == cointegrated_periods[0][0] else ""
                )
            
            # Calculate and display cointegration statistics
            pct_cointegrated = (rolling_pvalues < 0.05).mean() * 100
            
            fig_coint.update_layout(
                title=f"Rolling {st.session_state.settings['cointegration_window']}-Day Cointegration Analysis<br><sub>Strategy is statistically valid {pct_cointegrated:.1f}% of the time</sub>",
                xaxis_title="Date",
                yaxis_title="P-value (Lower = Better)",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0, max(0.15, rolling_pvalues.max() * 1.1)]),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_coint, use_container_width=True)
            
            # Cointegration summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <h4>Strategy Validity</h4>
                <p style="font-size: 2em; font-weight: bold; color: #6366f1;">{pct_cointegrated:.1f}%</p>
                <p>Time periods with statistical backing</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_pvalue = rolling_pvalues.mean()
                st.markdown(f"""
                <div class="metric-card">
                <h4>Average P-value</h4>
                <p style="font-size: 2em; font-weight: bold; color: #8b5cf6;">{avg_pvalue:.3f}</p>
                <p>Lower is better (< 0.05 target)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                current_pval = rolling_pvalues.iloc[-1]
                st.markdown(f"""
                <div class="metric-card">
                <h4>Current P-value</h4>
                <p style="font-size: 2em; font-weight: bold; color: {"#10b981" if current_pval < 0.05 else "#ef4444"};">{current_pval:.3f}</p>
                <p>{"✅ Significant" if current_pval < 0.05 else "❌ Not Significant"}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                strong_periods = (rolling_pvalues < 0.01).mean() * 100
                st.markdown(f"""
                <div class="metric-card">
                <h4>Strong Periods</h4>
                <p style="font-size: 2em; font-weight: bold; color: #10b981;">{strong_periods:.1f}%</p>
                <p>P-value < 0.01 (very strong)</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ Need at least {st.session_state.settings['cointegration_window']} data points for rolling cointegration analysis.")
    
    with tab2:
        st.header("📊 Statistical Metrics")
        
        # Calculate spread statistics
        ps_stats = calculate_spread_statistics(filtered_df['Point Spread'])
        as_stats = calculate_spread_statistics(filtered_df['Accum Spread'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Point Spread (PS) Statistics")
            st.markdown("""
            <div class="stats-table">
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td><b>Max</b></td><td>{:.6f}</td></tr>
                <tr><td><b>Min</b></td><td>{:.6f}</td></tr>
                <tr><td><b>Mean</b></td><td>{:.6f}</td></tr>
                <tr><td><b>Median</b></td><td>{:.6f}</td></tr>
                <tr><td><b>Std Dev</b></td><td>{:.6f}</td></tr>
                <tr><td><b>MAD</b></td><td>{:.6f}</td></tr>
                <tr style="background: rgba(239, 68, 68, 0.1);"><td><b>20th Percentile</b></td><td><b>{:.6f}</b></td></tr>
                <tr style="background: rgba(99, 102, 241, 0.2);"><td><b>50th Percentile (Median)</b></td><td><b>{:.6f}</b></td></tr>
                <tr style="background: rgba(239, 68, 68, 0.1);"><td><b>80th Percentile</b></td><td><b>{:.6f}</b></td></tr>
                <tr style="background: rgba(139, 92, 246, 0.3);"><td><b>MAD -2σ Threshold</b></td><td><b>{:.6f}</b></td></tr>
                <tr style="background: rgba(139, 92, 246, 0.3);"><td><b>MAD +2σ Threshold</b></td><td><b>{:.6f}</b></td></tr>
            </table>
            </div>
            """.format(
                ps_stats['max'], ps_stats['min'], ps_stats['mean'], ps_stats['median'], 
                ps_stats['std'], ps_stats['mad'], ps_stats['p20'], ps_stats['p50'], 
                ps_stats['p80'], ps_stats['mad_2_lower'], ps_stats['mad_2_upper']
            ), unsafe_allow_html=True)
        
        with col2:
            st.subheader("Accumulated Spread (AS) Statistics")
            st.markdown("""
            <div class="stats-table">
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td><b>Max</b></td><td>{:.6f}</td></tr>
                <tr><td><b>Min</b></td><td>{:.6f}</td></tr>
                <tr><td><b>Mean</b></td><td>{:.6f}</td></tr>
                <tr><td><b>Median</b></td><td>{:.6f}</td></tr>
                <tr><td><b>Std Dev</b></td><td>{:.6f}</td></tr>
                <tr><td><b>MAD</b></td><td>{:.6f}</td></tr>
                <tr style="background: rgba(239, 68, 68, 0.1);"><td><b>20th Percentile</b></td><td><b>{:.6f}</b></td></tr>
                <tr style="background: rgba(99, 102, 241, 0.2);"><td><b>50th Percentile (Median)</b></td><td><b>{:.6f}</b></td></tr>
                <tr style="background: rgba(239, 68, 68, 0.1);"><td><b>80th Percentile</b></td><td><b>{:.6f}</b></td></tr>
                <tr style="background: rgba(139, 92, 246, 0.3);"><td><b>MAD -2σ Threshold</b></td><td><b>{:.6f}</b></td></tr>
                <tr style="background: rgba(139, 92, 246, 0.3);"><td><b>MAD +2σ Threshold</b></td><td><b>{:.6f}</b></td></tr>
            </table>
            </div>
            """.format(
                as_stats['max'], as_stats['min'], as_stats['mean'], as_stats['median'], 
                as_stats['std'], as_stats['mad'], as_stats['p20'], as_stats['p50'], 
                as_stats['p80'], as_stats['mad_2_lower'], as_stats['mad_2_upper']
            ), unsafe_allow_html=True)
        
        # Distribution charts - Ricardo's Key Requirement
        st.subheader("📊 Spread Distributions (Bell Curve Analysis)")
        
        st.markdown("""
        <div class="warning-box">
        <b>Ricardo's Note:</b> MAD Z-scores should be visualized as bell curves, not linear charts. 
        This shows the true statistical distribution and helps identify entry/exit thresholds properly.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # MAD Z-score distribution as bell curve
            fig_dist = go.Figure()
            
            # Create histogram data
            z_scores = filtered_df['mad_z_score'].dropna()
            
            # Add histogram
            fig_dist.add_trace(go.Histogram(
                x=z_scores,
                nbinsx=50,
                name='MAD Z-Score Distribution',
                marker_color='#8b5cf6',
                opacity=0.7,
                histnorm='probability density'  # Normalize for proper bell curve overlay
            ))
            
            # Add normal distribution overlay
            x_range = np.linspace(z_scores.min(), z_scores.max(), 100)
            normal_dist = stats.norm.pdf(x_range, loc=z_scores.mean(), scale=z_scores.std())
            
            fig_dist.add_trace(go.Scatter(
                x=x_range,
                y=normal_dist,
                mode='lines',
                name='Normal Distribution Fit',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            # Add actual data distribution (KDE-like)
            from scipy.stats import gaussian_kde
            if len(z_scores) > 10:  # Need sufficient data for KDE
                kde = gaussian_kde(z_scores)
                kde_x = np.linspace(z_scores.min(), z_scores.max(), 100)
                kde_y = kde(kde_x)
                
                fig_dist.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode='lines',
                    name='Actual Distribution (KDE)',
                    line=dict(color='#6366f1', width=2)
                ))
            
            # Add threshold lines
            fig_dist.add_vline(x=-st.session_state.settings['mad_k_entry'], 
                             line_dash="solid", line_color="green", line_width=3,
                             annotation_text=f"Entry: -{st.session_state.settings['mad_k_entry']}σ")
            fig_dist.add_vline(x=st.session_state.settings['mad_k_exit'], 
                             line_dash="solid", line_color="red", line_width=3,
                             annotation_text=f"Exit: {st.session_state.settings['mad_k_exit']}σ (Median)")
            
            # Add current position
            current_z = z_scores.iloc[-1] if len(z_scores) > 0 else 0
            fig_dist.add_vline(x=current_z, 
                             line_dash="dot", line_color="orange", line_width=3,
                             annotation_text=f"Current: {current_z:.2f}σ")
            
            fig_dist.update_layout(
                title="MAD Z-Score Distribution (Bell Curve Analysis)",
                xaxis_title="MAD Z-Score (σ)",
                yaxis_title="Probability Density",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Q-Q plot for normality check
            fig_qq = go.Figure()
            
            # Calculate theoretical quantiles
            sorted_z = np.sort(z_scores)
            norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_z)))
            
            fig_qq.add_trace(go.Scatter(
                x=norm_quantiles,
                y=sorted_z,
                mode='markers',
                name='Data',
                marker=dict(color='#6366f1', size=5)
            ))
            
            # Add 45-degree line
            min_val = min(norm_quantiles.min(), sorted_z.min())
            max_val = max(norm_quantiles.max(), sorted_z.max())
            fig_qq.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal',
                line=dict(color='red', dash='dash')
            ))
            
            fig_qq.update_layout(
                title="Q-Q Plot (Normality Check)",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_qq, use_container_width=True)
    
    with tab3:
        st.header("📈 MAD-Based Trading Signals")
        
        st.markdown("""
        <div class="info-box">
        <b>Why MAD is Superior (Ricardo's Analysis):</b>
        <ul>
        <li><b>Robustness:</b> MAD is far less sensitive to one-off price shocks than mean ± σ or rolling percentiles</li>
        <li><b>Interpretability:</b> "You're > 2 MAD from median" reads like a true Z-score - easy to explain</li>
        <li><b>Consistency:</b> MAD rescales every pair to "median distance" - compare ETH/BCH and BTC/ADA without re-tuning</li>
        <li><b>Adaptability:</b> Percentile cut-offs drift when volatility regime shifts, MAD adapts automatically</li>
        </ul>
        <b>Trade-offs:</b> MAD assumes roughly symmetric spread distribution. For skewed spreads, we layer a percentile check.
        </div>
        """, unsafe_allow_html=True)
        
        # Main spread chart with MAD bands
        fig_mad = go.Figure()
        
        # Add log spread
        fig_mad.add_trace(go.Scatter(
            x=filtered_df['ISO Date'],
            y=filtered_df['log_spread'],
            name='Log Price Spread',
            line=dict(color='#1f2937', width=1)
        ))
        
        # Add median line
        fig_mad.add_trace(go.Scatter(
            x=filtered_df['ISO Date'],
            y=filtered_df['spread_median'],
            name='Rolling Median',
            line=dict(color='#6366f1', width=2)
        ))
        
        # Add MAD bands
        upper_band = filtered_df['spread_median'] + st.session_state.settings['mad_k_entry'] * filtered_df['spread_mad'] * 1.4826
        lower_band = filtered_df['spread_median'] - st.session_state.settings['mad_k_entry'] * filtered_df['spread_mad'] * 1.4826
        
        fig_mad.add_trace(go.Scatter(
            x=filtered_df['ISO Date'],
            y=upper_band,
            name=f'+{st.session_state.settings["mad_k_entry"]} MAD',
            line=dict(color='#ef4444', width=1, dash='dash')
        ))
        
        fig_mad.add_trace(go.Scatter(
            x=filtered_df['ISO Date'],
            y=lower_band,
            name=f'-{st.session_state.settings["mad_k_entry"]} MAD',
            line=dict(color='#10b981', width=1, dash='dash')
        ))
        
        # Add entry/exit signals
        entry_points = filtered_df[filtered_df['entry_signal'] == 1]
        exit_points = filtered_df[filtered_df['exit_signal'] == 1]
        
        if not entry_points.empty:
            fig_mad.add_trace(go.Scatter(
                x=entry_points['ISO Date'],
                y=entry_points['log_spread'],
                mode='markers',
                name='Entry Signal',
                marker=dict(color='#10b981', size=10, symbol='triangle-up')
            ))
        
        if not exit_points.empty:
            fig_mad.add_trace(go.Scatter(
                x=exit_points['ISO Date'],
                y=exit_points['log_spread'],
                mode='markers',
                name='Exit Signal',
                marker=dict(color='#ef4444', size=10, symbol='triangle-down')
            ))
        
        fig_mad.update_layout(
            title="Log Price Spread with MAD Bands & Trading Signals",
            xaxis_title="Date",
            yaxis_title="Log Price Spread (ln(ETH) - ln(BCH))",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_mad, use_container_width=True)
        
        # Signal summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_entries = len(entry_points)
            st.markdown(f"""
            <div class="metric-card">
            <h4>Entry Signals</h4>
            <p style="font-size: 2em; font-weight: bold; color: #10b981;">{total_entries}</p>
            <p>When spread < -{st.session_state.settings['mad_k_entry']} MAD</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_exits = len(exit_points)
            st.markdown(f"""
            <div class="metric-card">
            <h4>Exit Signals</h4>
            <p style="font-size: 2em; font-weight: bold; color: #ef4444;">{total_exits}</p>
            <p>When spread crosses median</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            latest_z = filtered_df['mad_z_score'].iloc[-1]
            signal_status = "ENTRY" if latest_z <= -st.session_state.settings['mad_k_entry'] else "EXIT" if latest_z >= 0 else "WAIT"
            color = "#10b981" if signal_status == "ENTRY" else "#ef4444" if signal_status == "EXIT" else "#f59e0b"
            
            st.markdown(f"""
            <div class="metric-card">
            <h4>Current Status</h4>
            <p style="font-size: 2em; font-weight: bold; color: {color};">{signal_status}</p>
            <p>Z-Score: {latest_z:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header("🎯 Trading Opportunities")
        
        # Calculate opportunity scores based on cointegration and risk metrics
        opportunities = []
        
        # Check different lookback periods
        lookback_periods = [30, 60, 120, 252]
        
        for period in lookback_periods:
            if len(filtered_df) < period:
                continue
            
            # Get recent data
            recent_df = filtered_df.tail(period)
            
            # Calculate metrics
            coint_result = calculate_cointegration(
                np.log(recent_df['ETH Close']), 
                np.log(recent_df['BCH Close'])
            )
            
            avg_corr = recent_df['Rolling Corr'].mean()
            avg_beta = recent_df['beta'].mean()
            current_z = recent_df['mad_z_score'].iloc[-1]
            
            # Calculate opportunity score (0-100) - IMPROVED SCORING LOGIC
            score = 0
            
            # Cointegration score (50 points max) - Most important factor
            if coint_result['pvalue'] < 0.01:
                score += 50  # Strong cointegration
            elif coint_result['pvalue'] < 0.05:
                score += 35  # Moderate cointegration
            elif coint_result['pvalue'] < 0.10:
                score += 20  # Weak cointegration
            else:
                score += 0   # No cointegration - major red flag
            
            # Risk filter score (25 points max)
            if avg_corr >= st.session_state.settings['correlation_threshold']:
                score += 12.5
            if st.session_state.settings['beta_min'] <= avg_beta <= st.session_state.settings['beta_max']:
                score += 12.5
            
            # Signal strength score (25 points max) - Based on MAD thresholds
            if current_z <= -st.session_state.settings['mad_k_entry']:
                score += 25  # Strong entry signal
            elif current_z <= -st.session_state.settings['mad_k_entry'] * 0.75:
                score += 18  # Moderate entry signal
            elif current_z <= -st.session_state.settings['mad_k_entry'] * 0.5:
                score += 10  # Weak entry signal
            else:
                score += 0   # No entry signal
            
            opportunities.append({
                'period': f"{period} days",
                'score': score,
                'cointegration_pvalue': coint_result['pvalue'],
                'is_cointegrated': coint_result['is_cointegrated'],
                'avg_correlation': avg_corr,
                'avg_beta': avg_beta,
                'current_z_score': current_z,
                'entry_ready': current_z <= -st.session_state.settings['mad_k_entry'],
                'cointegration_strength': coint_result['confidence']
            })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        # Display opportunities - ONLY SHOW MEANINGFUL OPPORTUNITIES
        st.subheader("Opportunity Ranking (Cointegration-Based)")
        
        st.markdown("""
        <div class="info-box">
        <b>Scoring Logic:</b>
        <ul>
        <li><b>Cointegration (50 pts):</b> P-value < 0.01 = 50pts, < 0.05 = 35pts, < 0.10 = 20pts</li>
        <li><b>Risk Filters (25 pts):</b> Correlation ≥ threshold + Beta in range</li>
        <li><b>Entry Signal (25 pts):</b> Current MAD Z-score relative to entry threshold</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Only show opportunities with cointegration (score > 20)
        meaningful_opportunities = [opp for opp in opportunities if opp['score'] > 20]
        
        if not meaningful_opportunities:
            st.warning("⚠️ No cointegrated opportunities found. Consider adjusting parameters or waiting for better market conditions.")
        else:
            for i, opp in enumerate(meaningful_opportunities):
                color = "#10b981" if opp['score'] >= 70 else "#f59e0b" if opp['score'] >= 50 else "#ef4444"
                status = "🟢 STRONG" if opp['score'] >= 70 else "🟡 MODERATE" if opp['score'] >= 50 else "🔴 WEAK"
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background: {color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {color};">
                    <h4>{opp['period']} Analysis</h4>
                    <p><b>Opportunity Score:</b> {opp['score']:.0f}/100 {status}</p>
                    <p><b>Cointegration:</b> {opp['cointegration_strength']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    coint_status = "✅" if opp['is_cointegrated'] else "❌"
                    st.markdown(f"""
                    <div class="info-box">
                    <p><b>Cointegration:</b> {coint_status}</p>
                    <p>P-value: {opp['cointegration_pvalue']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    risk_ok = (opp['avg_correlation'] >= st.session_state.settings['correlation_threshold'] and
                              st.session_state.settings['beta_min'] <= opp['avg_beta'] <= st.session_state.settings['beta_max'])
                    risk_status = "✅" if risk_ok else "❌"
                    st.markdown(f"""
                    <div class="info-box">
                    <p><b>Risk Filters:</b> {risk_status}</p>
                    <p>Corr: {opp['avg_correlation']:.3f}</p>
                    <p>Beta: {opp['avg_beta']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    entry_status = "✅ ENTRY" if opp['entry_ready'] else "⏳ WAIT"
                    entry_color = "#10b981" if opp['entry_ready'] else "#f59e0b"
                    st.markdown(f"""
                    <div class="info-box">
                    <p style="color: {entry_color}; font-weight: bold;">{entry_status}</p>
                    <p>Z-Score: {opp['current_z_score']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab5:
        st.header("📋 Position Management")
        
        # Position sizing based on volatility
        recent_volatility = filtered_df['Point Spread'].tail(30).std()
        suggested_size = min(
            st.session_state.settings['base_capital'],
            st.session_state.settings['base_capital'] * 0.02 / (2 * recent_volatility)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Position Sizing")
            st.markdown(f"""
            <div class="info-box">
            <h4>Volatility-Based Sizing</h4>
            <p><b>30-Day Volatility:</b> {recent_volatility:.4f}</p>
            <p><b>Base Capital:</b> ${st.session_state.settings['base_capital']:,}</p>
            <p><b>Suggested Position:</b> ${suggested_size:,.0f}</p>
            <p><b>Max LTV:</b> {st.session_state.settings['max_ltv']:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Exit Strategy")
            st.markdown(f"""
            <div class="success-box">
            <h4>📋 Clear Exit Rules (Ricardo's Method)</h4>
            <p><b>Primary Exit:</b> Median-Cross Rule - Exit when spread crosses above 50th percentile</p>
            <p><b>MAD Exit:</b> When MAD Z-score crosses above {st.session_state.settings['mad_k_exit']} (median)</p>
            <p><b>Risk Stop:</b> If correlation drops below {st.session_state.settings['correlation_threshold']:.2f}</p>
            <p><b>Beta Stop:</b> If beta moves outside {st.session_state.settings['beta_min']:.1f}-{st.session_state.settings['beta_max']:.1f} range</p>
            <p><b>Time Stop:</b> After {int(half_life * 2)} days (2x half-life)</p>
            <p><b>Cointegration Stop:</b> If daily p-value > 0.05 for 3+ days</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent signals table
        st.subheader("Recent Trading Signals")
        
        recent_signals = filtered_df[
            (filtered_df['entry_signal'] == 1) | (filtered_df['exit_signal'] == 1)
        ].tail(10)
        
        if not recent_signals.empty:
            display_df = recent_signals[[
                'ISO Date', 'ETH Close', 'BCH Close', 'log_spread', 
                'mad_z_score', 'Rolling Corr', 'beta', 'entry_signal', 'exit_signal'
            ]].copy()
            
            display_df['Signal'] = display_df.apply(
                lambda x: '🟢 ENTRY' if x['entry_signal'] == 1 else '🔴 EXIT', axis=1
            )
            
            display_df = display_df.drop(['entry_signal', 'exit_signal'], axis=1)
            
            st.dataframe(
                display_df.style.format({
                    'ETH Close': '${:.2f}',
                    'BCH Close': '${:.2f}',
                    'log_spread': '{:.4f}',
                    'mad_z_score': '{:.2f}',
                    'Rolling Corr': '{:.3f}',
                    'beta': '{:.3f}'
                }),
                use_container_width=True
            )
        else:
            st.info("No recent trading signals found.")

if __name__ == "__main__":
    main()
