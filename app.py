import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="Ricardo's Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize session state for settings ---
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'entry_ps_threshold': 0.05,
        'entry_as_threshold': 0.20,
        'exit_as_threshold': 0.50,
        'z_entry_threshold': -2.0,
        'z_exit_threshold': 0.5,
        'lookback_window': 120,
        'mad_window': 120,
        'correlation_threshold': 0.5,
        'beta_min': 1.0,
        'beta_max': 2.0,
        'max_ltv': 0.75,
        'target_apr_min': 0.08,
        'target_apr_max': 0.25,
        'max_drawdown': 0.30,
        'strong_coin': 'ETH',
        'weak_coin': 'BCH',
        'use_z_score_signals': False,
        'position_sizing_method': 'fixed',
        'base_capital': 10000,
        'risk_per_trade': 0.02
    }

# --- Custom CSS for a Professional Monochromatic Dark Theme ---
st.markdown("""
<style>
    :root {
        --bg-color: #0a0a0a;
        --bg-color-light: #1a1a1a;
        --card-bg-color: #1c1c1c;
        --border-color: #333333;
        --text-color: #ffffff;
        --text-secondary-color: #e5e5e5;
        --primary-color: #ffffff; 
        --accent-color: #f0f0f0;
        --success-color: #22c55e;
        --danger-color: #ef4444;
    }

    /* Global Styling */
    html, body, .main {
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Tab content text only */
    .stTabs [data-baseweb="tab-panel"] {
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] .stMarkdown,
    .stTabs [data-baseweb="tab-panel"] .stMarkdown p,
    .stTabs [data-baseweb="tab-panel"] .stMarkdown div,
    .stTabs [data-baseweb="tab-panel"] .stMarkdown span,
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] div,
    .stTabs [data-baseweb="tab-panel"] span {
        color: #ffffff !important;
    }
    
    /* Specific targeting for tab content elements */
    .stTabs [data-baseweb="tab-panel"] .element-container,
    .stTabs [data-baseweb="tab-panel"] .element-container * {
        color: #ffffff !important;
    }
    
    /* Strategy summary section text */
    .stMarkdown ul li, .stMarkdown ol li {
        color: #ffffff !important;
    }
    
    /* Chart container text */
    .chart-container * {
        color: #ffffff !important;
    }
    
    /* Info boxes within tabs */
    .stTabs .stInfo, .stTabs .stWarning, .stTabs .stError, .stTabs .stSuccess {
        color: #ffffff !important;
    }
    
    /* Dataframe text - keep dark for readability */
    .stDataFrame * {
        color: #000000 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    h2 {
        font-size: 1.75rem !important;
        color: #f8f9fa !important;
    }
    
    h3, h4 {
        color: #e9ecef !important;
        font-weight: 600 !important;
    }
    
    .stApp {
        background-image: linear-gradient(180deg, var(--bg-color-light) 0%, var(--bg-color) 300px);
    }

    /* Header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 2rem;
        background-color: transparent;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .header-container h1 {
        font-size: 2rem;
        letter-spacing: -1px;
    }

    /* Metric Cards */
    .metric-card {
        background-color: var(--card-bg-color);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 1rem;
        transition: all 0.3s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.4);
        border-color: #555;
    }

    .metric-label {
        font-size: 14px;
        color: #ffffff !important;
        margin-bottom: 8px;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff !important;
        line-height: 1.2;
    }
    
    .metric-value.positive {
        color: var(--success-color);
    }
    
    .metric-value.negative {
        color: var(--danger-color);
    }

    /* Chart Containers */
    .chart-container {
        background-color: var(--card-bg-color);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    
    .chart-container h3, .chart-container h4 {
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-color);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown div,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown strong,
    [data-testid="stSidebar"] .stMarkdown b {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stButton button {
        color: #ffffff !important;
    }
    
    /* Input field text should be black for readability */
    [data-testid="stSidebar"] .stSelectbox > div > div > div,
    [data-testid="stSidebar"] .stSelectbox input,
    [data-testid="stSidebar"] .stDateInput input,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Dropdown options */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
        color: #000000 !important;
    }
    
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        font-weight: 600;
        padding: 14px 18px;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--card-bg-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: #000 !important;
        box-shadow: 0 2px 10px rgba(255, 255, 255, 0.1);
    }

    /* Expander */
    .streamlit-expander {
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        background: var(--card-bg-color) !important;
        margin-bottom: 1.5rem !important;
    }

    .streamlit-expander > summary {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        background-color: var(--card-bg-color) !important;
        padding: 1rem !important;
    }
    
    .streamlit-expander > summary:hover {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }
    
    .streamlit-expander-body {
         background-color: var(--bg-color-light) !important;
         padding: 1rem !important;
    }
    
    /* Guide content styling */
    .guide-content {
        background-color: var(--bg-color-light);
        padding: 1.5rem;
        border-radius: 8px;
        color: #ffffff !important;
        line-height: 1.6;
    }
    
    .guide-content ul {
        margin: 1rem 0;
        padding-left: 1.5rem;
    }
    
    .guide-content li {
        margin-bottom: 0.5rem;
        color: #ffffff !important;
    }
    
    .guide-content b {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 12px;
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

@st.cache_data
def load_multi_timeframe_data():
    """Load all available timeframes from the Excel file"""
    try:
        # Read the Excel file to get all sheet names
        excel_file = pd.ExcelFile('ETH_BCH_multi_interval_analysis.xlsx')
        timeframe_data = {}
        
        # Filter sheets that contain timeframe data (exclude 'Resumen')
        timeframe_sheets = [sheet for sheet in excel_file.sheet_names if sheet != 'Resumen' and '_n=' in sheet]
        
        for sheet in timeframe_sheets:
            df = pd.read_excel('ETH_BCH_multi_interval_analysis.xlsx', sheet_name=sheet)
            df['ISO Date'] = pd.to_datetime(df['ISO Date'])
            
            # Check if dates are in the wrong range (2022-2023 instead of 2024-2025)
            min_date = df['ISO Date'].min()
            max_date = df['ISO Date'].max()
            
            # If dates are from 2022-2023, shift them to 2024-2025
            if min_date.year < 2024:
                years_to_add = 2024 - min_date.year
                df['ISO Date'] = df['ISO Date'] + pd.DateOffset(years=years_to_add)
            
            # Extract timeframe and sample size from sheet name
            parts = sheet.split('_n=')
            timeframe = parts[0]
            sample_size = int(parts[1])
            
            if timeframe not in timeframe_data:
                timeframe_data[timeframe] = {}
            timeframe_data[timeframe][sample_size] = df
            
        return timeframe_data
    except Exception as e:
        st.error(f"Error loading multi-timeframe data: {str(e)}")
        return {}

def calculate_mad_z_scores(series, window=120):
    """Calculate MAD-based Z-scores for robust statistical analysis"""
    mad_z_scores = []
    
    for i in range(len(series)):
        if i < window:
            mad_z_scores.append(np.nan)
            continue
            
        # Get rolling window
        window_data = series.iloc[i-window:i]
        
        # Calculate median
        median_val = window_data.median()
        
        # Calculate MAD (Median Absolute Deviation)
        mad = np.median(np.abs(window_data - median_val))
        
        # Convert to sigma-like scale
        sigma_equivalent = 1.4826 * mad
        
        # Calculate Z-score
        if sigma_equivalent > 0:
            z_score = (series.iloc[i] - median_val) / sigma_equivalent
        else:
            z_score = 0
            
        mad_z_scores.append(z_score)
    
    return pd.Series(mad_z_scores, index=series.index)

def simulate_cointegration_test(price1, price2, window_days=180):
    """Simulate cointegration test (simplified version)"""
    if len(price1) < window_days:
        return {'p_value': 1.0, 'test_stat': 0.0, 'is_cointegrated': False}
    
    recent_data = price1.tail(window_days), price2.tail(window_days)
    
    # Simple correlation-based proxy for cointegration
    correlation = np.corrcoef(recent_data[0], recent_data[1])[0, 1]
    
    # Simulate p-value based on correlation strength
    p_value = max(0.001, 1 - abs(correlation))
    
    return {
        'p_value': p_value,
        'correlation': correlation,
        'is_cointegrated': p_value < 0.05,
        'strength': 'Strong' if p_value < 0.01 else 'Moderate' if p_value < 0.05 else 'Weak'
    }

def calculate_volatility_sizing(weak_coin_returns, max_risk_pct=0.30, base_capital=10000):
    """Calculate position size based on volatility"""
    if len(weak_coin_returns) < 30:
        return base_capital * 0.5
    
    # Calculate daily volatility
    daily_vol = weak_coin_returns.std()
    
    # 2-sigma move
    two_sigma_move = 2 * daily_vol
    
    # Position size to limit risk to max_risk_pct of capital
    if two_sigma_move > 0:
        optimal_size = (base_capital * max_risk_pct) / two_sigma_move
        return min(optimal_size, base_capital)  # Cap at total capital
    else:
        return base_capital * 0.5

def calculate_carry_costs(days_held, borrow_apr=0.10, lend_apr=0.04, swap_fee=0.002):
    """Calculate realistic trading costs"""
    daily_borrow_cost = borrow_apr / 365
    daily_lend_yield = lend_apr / 365
    
    net_daily_cost = daily_borrow_cost - daily_lend_yield
    total_carry_cost = net_daily_cost * days_held
    total_trading_cost = swap_fee * 2  # Entry + exit
    
    return {
        'carry_cost_pct': total_carry_cost * 100,
        'trading_cost_pct': total_trading_cost * 100,
        'total_cost_pct': (total_carry_cost + total_trading_cost) * 100
    }

def calculate_strategy_metrics(df, window):
    """Calculate metrics for the strategy deep dive charts."""
    # Calculate Relative Spread (RS) based on log prices
    df['RS'] = df['LN ETH Close'] - df['LN BCH Close']
    
    # Calculate rolling statistics for RS
    df['RS_median'] = df['RS'].rolling(window=window).median()
    df['RS_std'] = df['RS'].rolling(window=window).std()
    df['RS_mean'] = df['RS'].rolling(window=window).mean()
    
    # Calculate bands
    df['RS_upper_1_std'] = df['RS_mean'] + df['RS_std']
    df['RS_lower_1_std'] = df['RS_mean'] - df['RS_std']
    df['RS_upper_2_std'] = df['RS_mean'] + (2 * df['RS_std'])
    df['RS_lower_2_std'] = df['RS_mean'] - (2 * df['RS_std'])
    
    # Calculate Z-Score
    df['Z_score'] = (df['RS'] - df['RS_mean']) / df['RS_std']
    
    return df

def apply_custom_signals(df, settings):
    """Apply custom entry/exit signals based on user settings"""
    df_custom = df.copy()
    
    if settings['use_z_score_signals']:
        # Use Z-score based signals
        df_custom['custom_entry_signal'] = (df_custom['Z_score'] <= settings['z_entry_threshold']).astype(int)
        df_custom['custom_exit_signal'] = (df_custom['Z_score'] >= settings['z_exit_threshold']).astype(int)
    else:
        # Use percentile based signals with custom thresholds
        window = settings['lookback_window']
        
        # Recalculate percentiles with custom window
        df_custom['PS_pct_custom'] = df_custom['Point Spread'].rolling(window=window).rank(pct=True)
        df_custom['AS_pct_custom'] = df_custom['Accum Spread'].rolling(window=window).rank(pct=True)
        
        # Apply custom thresholds
        entry_condition = (
            (df_custom['PS_pct_custom'] < settings['entry_ps_threshold']) & 
            (df_custom['AS_pct_custom'] < settings['entry_as_threshold'])
        )
        exit_condition = df_custom['AS_pct_custom'] > settings['exit_as_threshold']
        
        df_custom['custom_entry_signal'] = entry_condition.astype(int)
        df_custom['custom_exit_signal'] = exit_condition.astype(int)
    
    # Apply risk filters
    corr_ok = df_custom['Rolling Corr'] >= settings['correlation_threshold']
    beta_ok = (df_custom['Rolling Beta'] >= settings['beta_min']) & (df_custom['Rolling Beta'] <= settings['beta_max'])
    risk_ok = corr_ok & beta_ok
    
    # Only allow signals when risk filters pass
    df_custom['custom_entry_signal'] = (df_custom['custom_entry_signal'] & risk_ok).astype(int)
    
    return df_custom

def run_backtest(df, settings, initial_capital=10000):
    """Run a comprehensive backtest of the trading strategy"""
    df_bt = apply_custom_signals(df, settings)
    
    # Initialize backtest variables
    portfolio_value = []
    cash = initial_capital
    position = 0  # 0 = no position, 1 = long position
    entry_price = 0
    entry_date = None
    trades = []
    
    # Calculate position sizing
    if settings['position_sizing_method'] == 'volatility':
        # Use volatility-based sizing
        volatility = df_bt['Point Spread'].rolling(30).std()
        position_sizes = (initial_capital * settings['risk_per_trade']) / (2 * volatility)
        position_sizes = position_sizes.fillna(initial_capital * 0.5).clip(upper=initial_capital)
    else:
        # Fixed position sizing
        position_sizes = pd.Series([initial_capital * 0.8] * len(df_bt), index=df_bt.index)
    
    for i, row in df_bt.iterrows():
        current_value = cash
        
        if position == 0 and row['custom_entry_signal'] == 1:
            # Enter position
            position = 1
            entry_price = row['ETH Close'] / row['BCH Close']  # ETH/BCH ratio
            entry_date = row['ISO Date']
            position_size = position_sizes.iloc[i]
            cash = initial_capital - position_size
            
        elif position == 1:
            # Update portfolio value based on current ratio
            current_ratio = row['ETH Close'] / row['BCH Close']
            position_pnl = (current_ratio - entry_price) / entry_price
            current_value = cash + (position_sizes.iloc[i] * (1 + position_pnl))
            
            if row['custom_exit_signal'] == 1:
                # Exit position
                exit_price = current_ratio
                trade_return = (exit_price - entry_price) / entry_price
                trade_duration = (row['ISO Date'] - entry_date).days
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': row['ISO Date'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'duration_days': trade_duration,
                    'position_size': position_sizes.iloc[i]
                })
                
                cash = current_value
                position = 0
                entry_price = 0
                entry_date = None
        
        portfolio_value.append(current_value)
    
    # Create results DataFrame
    df_bt['portfolio_value'] = portfolio_value
    df_bt['strategy_return'] = (df_bt['portfolio_value'] / initial_capital - 1) * 100
    
    # Calculate buy-and-hold benchmark (ETH)
    df_bt['eth_return'] = (df_bt['ETH Close'] / df_bt['ETH Close'].iloc[0] - 1) * 100
    
    # Calculate performance metrics
    strategy_returns = df_bt['strategy_return'].pct_change().dropna()
    
    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
        sharpe_ratio = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
        max_drawdown = calculate_max_drawdown(df_bt['portfolio_value'])
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    total_return = df_bt['strategy_return'].iloc[-1]
    eth_total_return = df_bt['eth_return'].iloc[-1]
    
    win_rate = len([t for t in trades if t['return'] > 0]) / len(trades) if trades else 0
    avg_trade_return = np.mean([t['return'] for t in trades]) if trades else 0
    avg_trade_duration = np.mean([t['duration_days'] for t in trades]) if trades else 0
    
    performance_metrics = {
        'total_return': total_return,
        'eth_benchmark_return': eth_total_return,
        'excess_return': total_return - eth_total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'avg_trade_duration': avg_trade_duration
    }
    
    return df_bt, trades, performance_metrics

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from portfolio values"""
    peak = portfolio_values.expanding().max()
    drawdown = (portfolio_values - peak) / peak
    return abs(drawdown.min()) * 100

def get_available_coin_pairs():
    """Get list of available trading pairs"""
    return [
        ('ETH', 'BCH'),
        ('ETH', 'LTC'), 
        ('BTC', 'ETH'),
        ('BTC', 'BCH'),
        ('BTC', 'LTC'),
        ('ETH', 'ADA'),
        ('ETH', 'DOT'),
        ('BTC', 'ADA')
    ]

def analyze_timeframe_opportunities(timeframe_data):
    """Analyze current opportunities across all timeframes with swing trading weights"""
    opportunities = []
    
    # Define swing trading weights for different timeframes
    swing_weights = {
        '1d': 0.45,    # Primary decision timeframe
        '4h': 0.35,    # Entry/exit timing
        '1h': 0.20,    # Short-term confirmation
        '30m': 0.05,   # Noise reduction
        '15m': 0.03,
        '5m': 0.02,
        '1m': 0.01
    }
    
    for timeframe, datasets in timeframe_data.items():
        for sample_size, df in datasets.items():
            if len(df) == 0:
                continue
                
            # Get latest data point
            latest = df.iloc[-1]
            
            # Check if data is recent (within reasonable timeframe)
            days_old = (pd.Timestamp.now() - latest['ISO Date']).days
            
            # Calculate opportunity score
            entry_ready = latest.get('entry_signal', 0) == 1
            exit_ready = latest.get('exit_signal', 0) == 1
            
            # Risk filters
            corr_ok = latest.get('Rolling Corr', 0) >= 0.5
            beta_ok = 1.0 <= latest.get('Rolling Beta', 0) <= 2.0
            
            # Signal strength
            ps_pct = latest.get('PS_pct', 0.5)
            as_pct = latest.get('AS_pct', 0.5)
            
            # Calculate base opportunity score (0-100)
            base_score = 0
            if entry_ready and corr_ok and beta_ok:
                # Strong entry signal
                base_score = 90 - (ps_pct * 100) - (as_pct * 100)
                base_score = max(0, min(100, base_score))
            elif corr_ok and beta_ok:
                # Conditions met but no signal yet
                base_score = 30 + (50 - abs(ps_pct - 0.5) * 100) + (50 - abs(as_pct - 0.5) * 100)
                base_score = max(0, min(60, base_score))
            
            # Apply swing trading weight
            timeframe_weight = swing_weights.get(timeframe, 0.01)
            weighted_score = base_score * timeframe_weight * 100  # Scale for visibility
            
            # Calculate statistical significance based on sample size
            days_covered = len(df)
            significance_bonus = min(20, days_covered / 10)  # Bonus for more data
            
            # Determine timeframe priority for swing trading
            if timeframe == '1d':
                priority = "🥇 Primary"
                priority_rank = 1
            elif timeframe == '4h':
                priority = "🥈 Timing"
                priority_rank = 2
            elif timeframe == '1h':
                priority = "🥉 Confirmation"
                priority_rank = 3
            else:
                priority = "📊 Context"
                priority_rank = 4
            
            opportunities.append({
                'timeframe': timeframe,
                'sample_size': sample_size,
                'opportunity_score': round(base_score, 1),
                'weighted_score': round(weighted_score, 1),
                'timeframe_weight': timeframe_weight,
                'priority': priority,
                'priority_rank': priority_rank,
                'entry_signal': entry_ready,
                'exit_signal': exit_ready,
                'correlation': round(latest.get('Rolling Corr', 0), 3),
                'beta': round(latest.get('Rolling Beta', 0), 3),
                'ps_percentile': round(ps_pct * 100, 1),
                'as_percentile': round(as_pct * 100, 1),
                'days_old': days_old,
                'days_covered': days_covered,
                'significance_bonus': round(significance_bonus, 1),
                'latest_date': latest['ISO Date'].strftime('%Y-%m-%d'),
                'risk_ok': corr_ok and beta_ok
            })
    
    # Sort by priority rank first, then by weighted score
    opportunities.sort(key=lambda x: (x['priority_rank'], -x['weighted_score']))
    return opportunities

def main():
    st.markdown("""
    <div class="header-container">
        <h1>📈 Ricardo's Trading Strategy Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Load multi-timeframe data
    timeframe_data = load_multi_timeframe_data()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Timeframe selector
    if timeframe_data:
        available_timeframes = list(timeframe_data.keys())
        selected_timeframe = st.sidebar.selectbox(
            "📊 Select Timeframe",
            options=available_timeframes,
            index=0 if available_timeframes else 0,
            help="Choose the timeframe for analysis. Different timeframes may show different opportunities."
        )
        
        # Sample size selector for the chosen timeframe
        if selected_timeframe in timeframe_data:
            available_samples = list(timeframe_data[selected_timeframe].keys())
            selected_sample = st.sidebar.selectbox(
                "📈 Sample Size",
                options=available_samples,
                index=0,
                help="Larger samples provide more historical context but may be less responsive to recent changes."
            )
            
            # Use the selected timeframe data
            df = timeframe_data[selected_timeframe][selected_sample]
        
        # Multi-timeframe opportunity scanner
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Opportunity Scanner")
        
        if st.sidebar.button("🚀 Scan All Timeframes", help="Analyze opportunities across all available timeframes"):
            opportunities = analyze_timeframe_opportunities(timeframe_data)
            
            st.sidebar.markdown("### 🎯 Swing Trading Priorities")
            
            # Group by priority for better organization
            primary_opps = [opp for opp in opportunities if opp['priority_rank'] == 1]
            timing_opps = [opp for opp in opportunities if opp['priority_rank'] == 2]
            confirmation_opps = [opp for opp in opportunities if opp['priority_rank'] == 3]
            
            # Show primary timeframes first
            if primary_opps:
                st.sidebar.markdown("**🥇 Primary (Daily):**")
                for opp in primary_opps[:2]:
                    status_emoji = "🟢" if opp['entry_signal'] else "🟡" if opp['opportunity_score'] > 40 else "🔴"
                    risk_emoji = "✅" if opp['risk_ok'] else "⚠️"
                    
                    st.sidebar.markdown(f"""
                    {status_emoji} **{opp['timeframe']}** (n={opp['sample_size']}) {risk_emoji}  
                    Score: {opp['opportunity_score']}/100 | Weight: {opp['timeframe_weight']:.0%}  
                    PS: {opp['ps_percentile']}% | AS: {opp['as_percentile']}%
                    """)
            
            if timing_opps:
                st.sidebar.markdown("**🥈 Timing (4H):**")
                for opp in timing_opps[:2]:
                    status_emoji = "🟢" if opp['entry_signal'] else "🟡" if opp['opportunity_score'] > 40 else "🔴"
                    risk_emoji = "✅" if opp['risk_ok'] else "⚠️"
                    
                    st.sidebar.markdown(f"""
                    {status_emoji} **{opp['timeframe']}** (n={opp['sample_size']}) {risk_emoji}  
                    Score: {opp['opportunity_score']}/100 | Weight: {opp['timeframe_weight']:.0%}  
                    PS: {opp['ps_percentile']}% | AS: {opp['as_percentile']}%
                    """)
            
            if confirmation_opps:
                st.sidebar.markdown("**🥉 Confirmation (1H):**")
                for opp in confirmation_opps[:1]:
                    status_emoji = "🟢" if opp['entry_signal'] else "🟡" if opp['opportunity_score'] > 40 else "🔴"
                    risk_emoji = "✅" if opp['risk_ok'] else "⚠️"
                    
                    st.sidebar.markdown(f"""
                    {status_emoji} **{opp['timeframe']}** (n={opp['sample_size']}) {risk_emoji}  
                    Score: {opp['opportunity_score']}/100 | Weight: {opp['timeframe_weight']:.0%}  
                    PS: {opp['ps_percentile']}% | AS: {opp['as_percentile']}%
                    """)
    
    # Date range selector
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
    
    # Key metrics row
    st.subheader("Key Strategy Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        entry_signals = filtered_df['entry_signal'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Entry Signals</div>
            <div class="metric-value">{entry_signals}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        exit_signals = filtered_df['exit_signal'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Exit Signals</div>
            <div class="metric-value">{exit_signals}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_correlation = filtered_df['Rolling Corr'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Correlation</div>
            <div class="metric-value">{avg_correlation:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_beta = filtered_df['Rolling Beta'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Beta</div>
            <div class="metric-value">{avg_beta:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Main charts section
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Price Analysis", 
        "📊 Spread & Signals", 
        "⚡ Entry/Exit Points", 
        "🛡️ Risk Management", 
        "🔬 Advanced Analytics",
        "💡 Strategy Deep Dive",
        "⏰ Multi-Timeframe",
        "⚙️ Settings",
        "📊 Backtest"
    ])
    
    with tab1:
        with st.expander("📖 Guide: Understanding Price Analysis", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab provides a foundational view of the two assets' price movements, both in absolute terms and relative to each other.

            <b>Key Definitions:</b>
            <ul>
                <li><b>Price Levels:</b> The raw closing price of each asset (ETH and BCH) over time.</li>
                <li><b>Percentage Returns:</b> The daily percentage change in price for each asset, calculated from log returns.</li>
                <li><b>Cumulative Growth:</b> Shows the total percentage growth of each asset from the beginning of the selected time period. This is useful for comparing performance over time from a common starting point.</li>
            </ul>

            <b>Charts Explained:</b>
            <ul>
                <li><b>ETH vs BCH Price Comparison:</b> The top chart shows the absolute price, while the bottom shows daily volatility (returns). This helps you see the raw price trends and daily movements side-by-side.</li>
                <li><b>Relative Performance Comparison:</b> This chart normalizes the starting price of both assets to 0% growth, making it easy to see which asset is outperforming over the selected period. The gap between the lines is what the strategy aims to capitalize on.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>ETH vs BCH Price Comparison</h3>", unsafe_allow_html=True)
        
        # Price comparison chart
        fig_prices = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Levels', 'Percentage Returns (%)'),
            vertical_spacing=0.1
        )
        
        # Price levels
        fig_prices.add_trace(
            go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['ETH Close'], 
                      name='ETH Close', line=dict(color='#3b82f6')),
            row=1, col=1
        )
        fig_prices.add_trace(
            go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['BCH Close'], 
                      name='BCH Close', line=dict(color='#ff7f0e')),
            row=1, col=1
        )
        
        # Percentage returns (cumulative)
        fig_prices.add_trace(
            go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['LN ETH Var %'] * 100, 
                      name='ETH Returns (%)', line=dict(color='#3b82f6')),
            row=2, col=1
        )
        fig_prices.add_trace(
            go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['LN BCH Var %'] * 100, 
                      name='BCH Returns (%)', line=dict(color='#ff7f0e')),
            row=2, col=1
        )
        
        # Add zero line for reference
        fig_prices.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig_prices.update_layout(
            height=600, 
            title_text=None, 
            template="plotly_dark", 
            showlegend=False,
            xaxis=dict(
                title="Date",
                type="date",
                tickformat="%Y-%m-%d"
            ),
            xaxis2=dict(
                title="Date", 
                type="date",
                tickformat="%Y-%m-%d"
            )
        )
        st.plotly_chart(fig_prices, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>Relative Performance Comparison</h3>", unsafe_allow_html=True)
        
        # Calculate cumulative returns from first day for comparison
        eth_base = filtered_df['ETH Close'].iloc[0]
        bch_base = filtered_df['BCH Close'].iloc[0]
        
        eth_cumulative_return = ((filtered_df['ETH Close'] / eth_base) - 1) * 100
        bch_cumulative_return = ((filtered_df['BCH Close'] / bch_base) - 1) * 100
        
        fig_relative = go.Figure()
        
        fig_relative.add_trace(
            go.Scatter(x=filtered_df['ISO Date'], y=eth_cumulative_return,
                      name='ETH Growth (%)', line=dict(color='#3b82f6', width=2))
        )
        
        fig_relative.add_trace(
            go.Scatter(x=filtered_df['ISO Date'], y=bch_cumulative_return,
                      name='BCH Growth (%)', line=dict(color='#ff7f0e', width=2))
        )
        
        fig_relative.update_layout(
            title=None,
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400,
            hovermode='x unified',
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                type="date",
                tickformat="%Y-%m-%d"
            )
        )
        
        st.plotly_chart(fig_relative, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        with st.expander("📖 Guide: Understanding Spread Analysis", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab gets to the core of the strategy: analyzing the performance gap (spread) between the two assets.

            <b>Key Definitions:</b>
            <ul>
                <li><b>Point Spread (Daily Gap):</b> The difference in daily percentage returns between the strong asset (ETH) and the weak asset (BCH). A positive value means ETH outperformed that day.</li>
                <li><b>Accumulated Spread:</b> The running total of the Point Spreads over time. This shows the long-term drift or trend in the performance gap. Mean-reversion strategies bet that this value will return to zero.</li>
            </ul>

            <b>Charts Explained:</b>
            <ul>
                <li><b>Spread Analysis:</b> This chart visualizes both the daily (point) and cumulative (accumulated) spreads. It helps you identify when the weak asset has had a run of outperformance, causing the accumulated spread to dip significantly, which can signal a trading opportunity.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>Spread Analysis</h3>", unsafe_allow_html=True)
        
        fig_spread = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Point Spread (Daily Gap)', 'Accumulated Spread'),
            vertical_spacing=0.15
        )
        
        fig_spread.add_trace(go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['Point Spread'], name='Point Spread', line=dict(color='#10b981')), row=1, col=1)
        fig_spread.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        fig_spread.add_trace(go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['Accum Spread'], name='Accumulated Spread', line=dict(color='#8b5cf6')), row=2, col=1)
        fig_spread.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig_spread.update_layout(
            height=600, 
            title_text=None, 
            template="plotly_dark", 
            showlegend=False,
            xaxis=dict(type="date", tickformat="%Y-%m-%d"),
            xaxis2=dict(type="date", tickformat="%Y-%m-%d")
        )
        st.plotly_chart(fig_spread, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        with st.expander("📖 Guide: Understanding Entry & Exit Points", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab shows exactly when the strategy's rules would trigger an entry or exit, based on the percentile-based signal logic.

            <b>Key Definitions:</b>
            <ul>
                <li><b>Entry Signal (Green Triangle):</b> Marked when the Point Spread is below the 5th percentile AND the Accumulated Spread is below the 20th percentile of their rolling history. This indicates an extreme, short-term outperformance by the weak asset.</li>
                <li><b>Exit Signal (Red Triangle):</b> Marked when the Accumulated Spread recovers to above the 50th percentile, suggesting the performance gap has closed.</li>
            </ul>

            <b>Charts Explained:</b>
            <ul>
                <li><b>Trading Signals on Accumulated Spread:</b> This chart plots the entry and exit markers directly on the accumulated spread line. This provides a clear visual of the conditions under which trades are initiated and closed according to the defined logic.</li>
            </ul>

            <b>Trading Logic:</b>
            <ul>
                <li><b>Why These Conditions?</b> The combination ensures we enter when both short-term (Point Spread) and medium-term (Accumulated Spread) conditions favor the weak asset, maximizing mean reversion potential.</li>
                <li><b>Risk Management:</b> Exit at 50th percentile prevents holding too long and captures the mean reversion move efficiently.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>Trading Signals on Accumulated Spread</h3>", unsafe_allow_html=True)
        
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['Accum Spread'], name='Accum. Spread', line=dict(color='#4b5563', width=1)))
        
        entry_points = filtered_df[filtered_df['entry_signal'] == 1]
        exit_points = filtered_df[filtered_df['exit_signal'] == 1]
        
        if not entry_points.empty:
            fig_signals.add_trace(go.Scatter(x=entry_points['ISO Date'], y=entry_points['Accum Spread'], mode='markers', name='Entry', marker=dict(color='var(--success-color)', size=10, symbol='triangle-up')))
        if not exit_points.empty:
            fig_signals.add_trace(go.Scatter(x=exit_points['ISO Date'], y=exit_points['Accum Spread'], mode='markers', name='Exit', marker=dict(color='var(--danger-color)', size=10, symbol='triangle-down')))

        fig_signals.update_layout(
            title=None, 
            height=500, 
            template="plotly_dark", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(type="date", tickformat="%Y-%m-%d")
        )
        st.plotly_chart(fig_signals, use_container_width=True)
        
        if not entry_points.empty:
            st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>Recent Entry Signals</h4>", unsafe_allow_html=True)
            st.dataframe(entry_points.tail(5)[['ISO Date', 'Point Spread', 'Accum Spread', 'PS_pct', 'AS_pct']], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        with st.expander("📖 Guide: Understanding Risk Management", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab displays the key statistical filters used to ensure a trading pair is suitable for the strategy, helping to manage risk.

            <b>Key Definitions:</b>
            <ul>
                <li><b>Rolling Correlation:</b> Measures how often the two assets move in the same direction. A value above 0.5 is required, ensuring the pair has a coherent relationship.</li>
                <li><b>Rolling Beta:</b> Measures the volatility of the weak asset (BCH) relative to the strong asset (ETH). A beta between 1 and 2 is required to ensure the weak asset isn't excessively volatile, which would increase liquidation risk.</li>
            </ul>

            <b>Charts Explained:</b>
            <ul>
                <li><b>Risk Management Metrics:</b> This chart tracks the rolling correlation and beta over time against their required thresholds (the white dashed lines). This allows you to see if the pair's relationship is stable and within the strategy's risk parameters.</li>
            </ul>

            <b>Why These Thresholds Matter:</b>
            <ul>
                <li><b>Correlation ≥ 0.5:</b> Ensures the assets move together enough for mean-reversion to work. Lower correlation means unpredictable relative movements.</li>
                <li><b>Beta ∈ [1,2]:</b> Beta < 1 means BCH is less volatile than ETH (reduces profit potential). Beta > 2 means excessive volatility increases liquidation risk when using leverage.</li>
                <li><b>Risk Control:</b> If either metric falls outside these ranges, the strategy should avoid new positions until conditions improve.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>Risk Management Metrics</h3>", unsafe_allow_html=True)
        
        fig_risk = make_subplots(rows=2, cols=1, subplot_titles=('Rolling Correlation', 'Rolling Beta'), vertical_spacing=0.15)
        
        fig_risk.add_trace(go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['Rolling Corr'], name='Correlation', line=dict(color='#3b82f6')), row=1, col=1)
        fig_risk.add_hline(y=0.5, line_dash="dash", line_color="white", row=1, col=1)
        
        fig_risk.add_trace(go.Scatter(x=filtered_df['ISO Date'], y=filtered_df['Rolling Beta'], name='Beta', line=dict(color='#eab308')), row=2, col=1)
        fig_risk.add_hline(y=1, line_dash="dash", line_color="white", row=2, col=1)
        fig_risk.add_hline(y=2, line_dash="dash", line_color="white", row=2, col=1)
        
        fig_risk.update_layout(
            height=600, 
            title_text=None, 
            template="plotly_dark", 
            showlegend=False,
            xaxis=dict(type="date", tickformat="%Y-%m-%d"),
            xaxis2=dict(type="date", tickformat="%Y-%m-%d")
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab5:
        with st.expander("📖 Guide: Understanding Advanced Analytics", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab explores more robust statistical methods to refine the trading signals and manage risk.

            <b>Key Definitions:</b>
            <ul>
                <li><b>MAD-Z Score:</b> A robust version of the standard Z-score. It uses the median instead of the mean, making it less sensitive to extreme outliers. It measures how many median absolute deviations (MAD) an observation is from the median. A score of +2 or -2 is generally considered statistically significant.</li>
                <li><b>Dual-Horizon Cointegration:</b> Tests both long-term structural relationships and short-term tactical opportunities between the asset pair.</li>
                <li><b>Volatility Sizing:</b> Dynamically adjusts position size based on recent volatility to maintain consistent risk exposure.</li>
                <li><b>Carry Costs:</b> Realistic trading costs including borrowing rates, lending yields, and swap fees that affect strategy profitability.</li>
            </ul>

            <b>Charts Explained:</b>
            <ul>
                <li><b>MAD-Z Score Analysis:</b> This chart plots the robust Z-score of the point spread over time. It provides an alternative, and often more stable, way to identify extreme deviations that could signal trading opportunities. The entry/exit thresholds provide a more statistically grounded way to define signals compared to percentiles.</li>
            </ul>

            <b>Advanced Features:</b>
            <ul>
                <li><b>Statistical Robustness:</b> MAD-Z scores are less affected by market outliers, providing more reliable signals during volatile periods.</li>
                <li><b>Dynamic Risk Management:</b> Volatility sizing ensures consistent risk exposure regardless of market conditions.</li>
                <li><b>Realistic Cost Modeling:</b> Carry cost analysis helps determine minimum profitable trade thresholds.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>Advanced Analytics</h3>", unsafe_allow_html=True)
        
        st.sidebar.header("🔧 Advanced Settings")
        mad_window = st.sidebar.slider("MAD-Z Window (days)", 60, 180, 120)
        
        if len(filtered_df) > mad_window:
            st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>MAD-Z Score Analysis</h4>", unsafe_allow_html=True)
            point_spread = filtered_df['Point Spread']
            mad_z_scores = calculate_mad_z_scores(point_spread, window=mad_window)
            
            fig_mad = go.Figure()
            fig_mad.add_trace(go.Scatter(x=filtered_df['ISO Date'], y=mad_z_scores, name='MAD-Z Score', line=dict(color='#9333ea')))
            fig_mad.add_hline(y=2.0, line_dash="dot", line_color='var(--danger-color)', annotation_text="Entry Threshold (+2σ)")
            fig_mad.add_hline(y=0.5, line_dash="dot", line_color='var(--success-color)', annotation_text="Exit Threshold (+0.5σ)")
            
            fig_mad.update_layout(
                height=500, 
                title_text=None, 
                template="plotly_dark",
                xaxis=dict(type="date", tickformat="%Y-%m-%d")
            )
            st.plotly_chart(fig_mad, use_container_width=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    with tab6:
        with st.expander("📖 Guide: Understanding Strategy Deep Dive", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab provides a granular look at the raw price spread and its statistical properties, offering the most detailed view of your strategy's mechanics.

            <b>Key Definitions:</b>
            <ul>
                <li><b>Price Spread (RS):</b> The raw difference between the log prices of the two assets (ln(ETH) - ln(BCH)). This is the underlying series for the strategy.</li>
                <li><b>Statistical Bands (±1σ, ±2σ):</b> These are the rolling standard deviations from the rolling mean of the price spread. They create a channel that helps visualize volatility and identify extreme deviations.</li>
                <li><b>Z-Score:</b> Measures how many standard deviations a data point is from the mean. It normalizes the spread, making it easy to compare across different volatility regimes.</li>
            </ul>

            <b>Charts Explained:</b>
            <ul>
                <li><b>Price Spread with Robust Bands:</b> This chart shows the raw spread with its statistical bands. Entry and exit signals are plotted to show where they fire relative to these bands.</li>
                <li><b>Z-Score with Entry/Exit Signals:</b> This chart plots the Z-score and shows where the percentile-based signals trigger. This is critical for assessing if your percentile thresholds align with statistically significant deviations (e.g., a Z-score of +/- 2).</li>
                <li><b>Histogram of Z-Scores:</b> This shows the distribution of Z-scores. A normal (bell-shaped) distribution is expected for a stable, mean-reverting pair.</li>
            </ul>

            <b>Strategic Insights:</b>
            <ul>
                <li><b>Signal Quality Assessment:</b> Compare where your entry signals fire on the Z-score chart. Ideally, entries should occur near -2σ for statistical significance.</li>
                <li><b>Mean Reversion Validation:</b> The histogram should show a normal distribution centered around zero, confirming the pair's mean-reverting behavior.</li>
                <li><b>Band Trading:</b> The statistical bands provide visual confirmation of when spreads are statistically extreme, validating your percentile-based approach.</li>
                <li><b>Strategy Optimization:</b> Use this data to fine-tune entry/exit thresholds based on actual Z-score distributions rather than just percentiles.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>💡 Strategy Deep Dive</h3>", unsafe_allow_html=True)
        st.markdown("Visualizations based on the raw price spread (`ln(ETH) - ln(BCH)`) to provide a deeper insight into your strategy's signals.")

        # Use the same window as the MAD-Z for consistency, or a dedicated one
        dive_window = st.sidebar.slider("Deep Dive Window (days)", 30, 200, 120, key="dive_window")

        if len(filtered_df) > dive_window:
            strategy_df = calculate_strategy_metrics(filtered_df.copy(), window=dive_window)

            # Chart 1: RS Spread with Robust Bands and Entry/Exit Marks
            st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>Price Spread with Robust Bands & Signals</h4>", unsafe_allow_html=True)
            fig_rs = go.Figure()

            # Add spread and bands
            fig_rs.add_trace(go.Scatter(x=strategy_df['ISO Date'], y=strategy_df['RS'], name='Price Spread', line=dict(color='orange')))
            fig_rs.add_trace(go.Scatter(x=strategy_df['ISO Date'], y=strategy_df['RS_mean'], name='Mean Spread', line=dict(color='red', dash='dash')))
            fig_rs.add_trace(go.Scatter(x=strategy_df['ISO Date'], y=strategy_df['RS_upper_1_std'], name='+1σ', line=dict(color='pink', width=1, dash='dot')))
            fig_rs.add_trace(go.Scatter(x=strategy_df['ISO Date'], y=strategy_df['RS_lower_1_std'], name='-1σ', line=dict(color='pink', width=1, dash='dot')))
            fig_rs.add_trace(go.Scatter(x=strategy_df['ISO Date'], y=strategy_df['RS_upper_2_std'], name='+2σ', line=dict(color='cyan', width=1, dash='dot')))
            fig_rs.add_trace(go.Scatter(x=strategy_df['ISO Date'], y=strategy_df['RS_lower_2_std'], name='-2σ', line=dict(color='cyan', width=1, dash='dot')))
            
            # Add entry/exit signals
            entry_points = strategy_df[strategy_df['entry_signal'] == 1]
            exit_points = strategy_df[strategy_df['exit_signal'] == 1]
            fig_rs.add_trace(go.Scatter(x=entry_points['ISO Date'], y=entry_points['RS'], mode='markers', name='Entry', marker=dict(color='green', symbol='triangle-up', size=10)))
            fig_rs.add_trace(go.Scatter(x=exit_points['ISO Date'], y=exit_points['RS'], mode='markers', name='Exit', marker=dict(color='red', symbol='triangle-down', size=10)))

            fig_rs.update_layout(
                title_text='Price Spread (ln ETH - ln BCH) with Statistical Bands', 
                template='plotly_dark', 
                height=500, 
                yaxis_title="Price Spread",
                xaxis=dict(type="date", tickformat="%Y-%m-%d")
            )
            st.plotly_chart(fig_rs, use_container_width=True)

            # Chart 2: Z-Score with Entry/Exit Thresholds
            st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>Z-Score with Entry/Exit Signals</h4>", unsafe_allow_html=True)
            fig_z = go.Figure()
            
            fig_z.add_trace(go.Scatter(x=strategy_df['ISO Date'], y=strategy_df['Z_score'], name='Z-Score', line=dict(color='orange')))
            fig_z.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Upper Threshold (+2σ)")
            fig_z.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="Lower Threshold (-2σ)")
            fig_z.add_hline(y=0, line_dash="dot", line_color="gray")

            # Add entry/exit signals
            fig_z.add_trace(go.Scatter(x=entry_points['ISO Date'], y=entry_points['Z_score'], mode='markers', name='Entry', marker=dict(color='green', symbol='triangle-up', size=10)))
            fig_z.add_trace(go.Scatter(x=exit_points['ISO Date'], y=exit_points['Z_score'], mode='markers', name='Exit', marker=dict(color='red', symbol='triangle-down', size=10)))

            fig_z.update_layout(title_text='Z-Score with Entry/Exit Signals', template='plotly_dark', height=400, yaxis_title="Z-Score")
            st.plotly_chart(fig_z, use_container_width=True)
            st.info("The Entry/Exit markers show at which Z-score level your current percentile-based signals are firing.")

            # Chart 3: Histogram of Z-Scores
            st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>Histogram of Z-Scores</h4>", unsafe_allow_html=True)
            fig_hist = px.histogram(strategy_df, x='Z_score', nbins=100, title='Z-Score Distribution', template='plotly_dark')
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        else:
            st.warning(f"Not enough data to display Deep Dive analytics. Please select a date range with at least {dive_window} days.")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab7:
        with st.expander("📖 Guide: Understanding Multi-Timeframe Analysis", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab helps you identify trading opportunities across different timeframes, as what appears as noise on one timeframe might be a clear signal on another.

            <b>Key Concepts:</b>
            <ul>
                <li><b>Timeframe Divergence:</b> Different timeframes can show completely different signals. A 1-hour chart might show no opportunity while a daily chart shows a strong entry signal.</li>
                <li><b>Sample Size Impact:</b> Larger sample sizes provide more historical context but may be slower to react to recent changes. Smaller samples are more responsive but may generate false signals.</li>
                <li><b>Opportunity Score:</b> A 0-100 score combining signal strength, risk filters, and percentile positions. Higher scores indicate better opportunities.</li>
                <li><b>Multi-Horizon Strategy:</b> Professional traders often use multiple timeframes to confirm signals and optimize entry/exit timing.</li>
            </ul>

            <b>How to Use:</b>
            <ul>
                <li><b>Opportunity Scanner:</b> Click "Scan All Timeframes" in the sidebar to see ranked opportunities across all available timeframes.</li>
                <li><b>Timeframe Comparison:</b> Compare the same metrics across different timeframes to understand market structure at various scales.</li>
                <li><b>Signal Confirmation:</b> Look for alignment between timeframes - when multiple timeframes show similar signals, confidence increases.</li>
                <li><b>Risk Assessment:</b> Ensure risk filters (correlation ≥ 0.5, beta ∈ [1,2]) are met across your chosen timeframes.</li>
            </ul>

            <b>Trading Strategy:</b>
            <ul>
                <li><b>Primary Timeframe:</b> Choose your main trading timeframe based on your holding period preference.</li>
                <li><b>Confirmation Timeframe:</b> Use a higher timeframe to confirm the overall trend direction.</li>
                <li><b>Entry Timing:</b> Use a lower timeframe to fine-tune your entry and exit points.</li>
                <li><b>Risk Management:</b> Monitor risk metrics across all relevant timeframes to avoid surprises.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>⏰ Multi-Timeframe Opportunity Analysis</h3>", unsafe_allow_html=True)
        
        if timeframe_data:
            # Generate comprehensive opportunity analysis
            opportunities = analyze_timeframe_opportunities(timeframe_data)
            
            if opportunities:
                # Create opportunity summary table
                st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>🎯 Opportunity Rankings</h4>", unsafe_allow_html=True)
                
                # Convert to DataFrame for better display
                opp_df = pd.DataFrame(opportunities)
                opp_df['Status'] = opp_df.apply(lambda row: 
                    '🟢 ENTRY READY' if row['entry_signal'] else 
                    '🟡 EXIT SIGNAL' if row['exit_signal'] else 
                    '🔴 NO SIGNAL', axis=1)
                
                # Display formatted table with swing trading context
                display_df = opp_df[['priority', 'timeframe', 'sample_size', 'opportunity_score', 'weighted_score', 
                                   'timeframe_weight', 'Status', 'correlation', 'beta', 'days_covered', 'ps_percentile', 'as_percentile']].copy()
                display_df.columns = ['Priority', 'Timeframe', 'Sample Size', 'Raw Score', 'Weighted Score', 'Weight', 'Status', 'Correlation', 'Beta', 'Days Data', 'PS %ile', 'AS %ile']
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualize opportunity scores
                st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>📊 Opportunity Score Comparison</h4>", unsafe_allow_html=True)
                
                fig_opp = go.Figure()
                
                # Create labels for x-axis
                labels = [f"{opp['timeframe']}_n{opp['sample_size']}" for opp in opportunities]
                scores = [opp['opportunity_score'] for opp in opportunities]
                colors = ['#22c55e' if opp['entry_signal'] else '#eab308' if opp['opportunity_score'] > 40 else '#ef4444' for opp in opportunities]
                
                fig_opp.add_trace(go.Bar(
                    x=labels,
                    y=scores,
                    marker_color=colors,
                    text=[f"{score}%" for score in scores],
                    textposition='auto',
                    name='Opportunity Score'
                ))
                
                fig_opp.update_layout(
                    title="Opportunity Scores Across All Timeframes",
                    xaxis_title="Timeframe & Sample Size",
                    yaxis_title="Opportunity Score (%)",
                    height=500,
                    template="plotly_dark",
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_opp, use_container_width=True)
                
                # Show detailed analysis for top opportunities
                st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>🔍 Top 3 Opportunities Detailed Analysis</h4>", unsafe_allow_html=True)
                
                for i, opp in enumerate(opportunities[:3]):
                    with st.expander(f"#{i+1}: {opp['timeframe']} (n={opp['sample_size']}) - Score: {opp['opportunity_score']}/100"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            **📊 Signal Status:**
                            - Entry Signal: {'✅' if opp['entry_signal'] else '❌'}
                            - Exit Signal: {'✅' if opp['exit_signal'] else '❌'}
                            - Risk Filters: {'✅' if opp['risk_ok'] else '❌'}
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **📈 Metrics:**
                            - Correlation: {opp['correlation']}
                            - Beta: {opp['beta']}
                            - Data Age: {opp['days_old']} days
                            """)
                        
                        with col3:
                            st.markdown(f"""
                            **🎯 Percentiles:**
                            - Point Spread: {opp['ps_percentile']}%
                            - Accumulated Spread: {opp['as_percentile']}%
                            - Latest Date: {opp['latest_date']}
                            """)
                
                # Timeframe correlation matrix
                st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>🔗 Cross-Timeframe Signal Correlation</h4>", unsafe_allow_html=True)
                
                # Create a matrix showing signal alignment across timeframes
                timeframes = list(set([opp['timeframe'] for opp in opportunities]))
                signal_matrix = []
                
                for tf1 in timeframes:
                    row = []
                    tf1_signals = [opp for opp in opportunities if opp['timeframe'] == tf1]
                    
                    for tf2 in timeframes:
                        tf2_signals = [opp for opp in opportunities if opp['timeframe'] == tf2]
                        
                        # Calculate alignment score (simplified)
                        if tf1_signals and tf2_signals:
                            avg_score_1 = np.mean([opp['opportunity_score'] for opp in tf1_signals])
                            avg_score_2 = np.mean([opp['opportunity_score'] for opp in tf2_signals])
                            alignment = 100 - abs(avg_score_1 - avg_score_2)
                        else:
                            alignment = 0
                        
                        row.append(alignment)
                    signal_matrix.append(row)
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=signal_matrix,
                    x=timeframes,
                    y=timeframes,
                    colorscale='RdYlGn',
                    text=[[f"{val:.1f}" for val in row] for row in signal_matrix],
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    hoverongaps=False
                ))
                
                fig_corr.update_layout(
                    title="Signal Alignment Between Timeframes (Higher = More Aligned)",
                    height=400,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Swing Trading Recommendations
                st.markdown("<h4 style='color: #ffffff !important; font-weight: 600;'>🎯 Swing Trading Recommendations</h4>", unsafe_allow_html=True)
                
                # Calculate consensus across primary timeframes
                primary_signals = [opp for opp in opportunities if opp['priority_rank'] <= 2]  # 1D and 4H
                
                if primary_signals:
                    entry_consensus = sum([opp['entry_signal'] for opp in primary_signals]) / len(primary_signals)
                    avg_score = np.mean([opp['opportunity_score'] for opp in primary_signals])
                    risk_consensus = all([opp['risk_ok'] for opp in primary_signals])
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        consensus_color = "#22c55e" if entry_consensus > 0.5 else "#eab308" if entry_consensus > 0 else "#ef4444"
                        st.markdown(f"""
                        <div style="background: {consensus_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {consensus_color};">
                        <h5 style="color: {consensus_color}; margin: 0;">Signal Consensus</h5>
                        <p style="color: #ffffff; margin: 0.5rem 0 0 0;">{entry_consensus:.0%} of primary timeframes show entry signals</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        score_color = "#22c55e" if avg_score > 60 else "#eab308" if avg_score > 30 else "#ef4444"
                        st.markdown(f"""
                        <div style="background: {score_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {score_color};">
                        <h5 style="color: {score_color}; margin: 0;">Average Score</h5>
                        <p style="color: #ffffff; margin: 0.5rem 0 0 0;">{avg_score:.1f}/100 across primary timeframes</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        risk_color = "#22c55e" if risk_consensus else "#ef4444"
                        st.markdown(f"""
                        <div style="background: {risk_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {risk_color};">
                        <h5 style="color: {risk_color}; margin: 0;">Risk Status</h5>
                        <p style="color: #ffffff; margin: 0.5rem 0 0 0;">{"All filters passed" if risk_consensus else "Risk filters failed"}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Trading recommendation
                    if entry_consensus > 0.5 and risk_consensus and avg_score > 50:
                        recommendation = "🟢 **STRONG BUY** - Multiple timeframes align with good risk profile"
                        rec_color = "#22c55e"
                    elif entry_consensus > 0 and risk_consensus and avg_score > 30:
                        recommendation = "🟡 **CAUTIOUS BUY** - Some signals present, monitor closely"
                        rec_color = "#eab308"
                    elif not risk_consensus:
                        recommendation = "🔴 **AVOID** - Risk filters not met, wait for better conditions"
                        rec_color = "#ef4444"
                    else:
                        recommendation = "⚪ **WAIT** - No clear signals, continue monitoring"
                        rec_color = "#6b7280"
                    
                    st.markdown(f"""
                    <div style="background: {rec_color}20; padding: 1.5rem; border-radius: 12px; border: 2px solid {rec_color}; margin-top: 1rem;">
                    <h4 style="color: {rec_color}; margin: 0 0 0.5rem 0;">Swing Trading Recommendation</h4>
                    <p style="color: #ffffff; margin: 0; font-size: 1.1rem;">{recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.warning("No opportunity data available. Please check your data files.")
        else:
            st.error("Multi-timeframe data not available. Please ensure the Excel file contains multiple timeframe sheets.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab8:
        with st.expander("📖 Guide: Understanding Settings", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab allows you to customize all strategy parameters and turn the dashboard into your personal research lab.

            <b>Signal Parameters:</b>
            <ul>
                <li><b>Entry/Exit Thresholds:</b> Adjust percentile or Z-score thresholds for entry and exit signals.</li>
                <li><b>Lookback Windows:</b> Control how much historical data is used for calculations.</li>
                <li><b>Signal Type:</b> Choose between percentile-based or Z-score-based signals.</li>
            </ul>

            <b>Risk Management:</b>
            <ul>
                <li><b>Correlation/Beta Filters:</b> Set minimum correlation and beta range requirements.</li>
                <li><b>Position Sizing:</b> Choose between fixed or volatility-based position sizing.</li>
                <li><b>Risk Limits:</b> Set maximum LTV, target APR ranges, and drawdown limits.</li>
            </ul>

            <b>Asset Selection:</b>
            <ul>
                <li><b>Trading Pairs:</b> Select different strong/weak coin combinations.</li>
                <li><b>Custom Parameters:</b> Tailor the strategy to different market conditions.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>⚙️ Strategy Settings & Parameters</h3>", unsafe_allow_html=True)
        
        # Create settings columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🎯 Signal Parameters")
            
            signal_type = st.radio(
                "Signal Type",
                ["Percentile-based", "Z-score-based"],
                index=0 if not st.session_state.settings['use_z_score_signals'] else 1,
                help="Choose between percentile or Z-score based entry/exit signals"
            )
            st.session_state.settings['use_z_score_signals'] = (signal_type == "Z-score-based")
            
            if st.session_state.settings['use_z_score_signals']:
                st.session_state.settings['z_entry_threshold'] = st.slider(
                    "Z-Score Entry Threshold", -3.0, 0.0, st.session_state.settings['z_entry_threshold'], 0.1,
                    help="Enter when Z-score falls below this level"
                )
                st.session_state.settings['z_exit_threshold'] = st.slider(
                    "Z-Score Exit Threshold", 0.0, 3.0, st.session_state.settings['z_exit_threshold'], 0.1,
                    help="Exit when Z-score rises above this level"
                )
            else:
                st.session_state.settings['entry_ps_threshold'] = st.slider(
                    "Point Spread Entry %ile", 0.01, 0.20, st.session_state.settings['entry_ps_threshold'], 0.01,
                    help="Enter when Point Spread percentile is below this level"
                )
                st.session_state.settings['entry_as_threshold'] = st.slider(
                    "Accumulated Spread Entry %ile", 0.05, 0.50, st.session_state.settings['entry_as_threshold'], 0.05,
                    help="Enter when Accumulated Spread percentile is below this level"
                )
                st.session_state.settings['exit_as_threshold'] = st.slider(
                    "Accumulated Spread Exit %ile", 0.30, 0.80, st.session_state.settings['exit_as_threshold'], 0.05,
                    help="Exit when Accumulated Spread percentile rises above this level"
                )
            
            st.session_state.settings['lookback_window'] = st.slider(
                "Lookback Window (days)", 30, 250, st.session_state.settings['lookback_window'], 10,
                help="Number of days to use for rolling calculations"
            )
            
            st.session_state.settings['mad_window'] = st.slider(
                "MAD-Z Window (days)", 30, 200, st.session_state.settings['mad_window'], 10,
                help="Window for MAD-Z score calculations"
            )
        
        with col2:
            st.markdown("#### 🛡️ Risk Management")
            
            st.session_state.settings['correlation_threshold'] = st.slider(
                "Min Correlation", 0.0, 1.0, st.session_state.settings['correlation_threshold'], 0.05,
                help="Minimum correlation required between assets"
            )
            
            st.session_state.settings['beta_min'] = st.slider(
                "Min Beta", 0.5, 2.0, st.session_state.settings['beta_min'], 0.1,
                help="Minimum beta (volatility ratio) allowed"
            )
            
            st.session_state.settings['beta_max'] = st.slider(
                "Max Beta", 1.0, 3.0, st.session_state.settings['beta_max'], 0.1,
                help="Maximum beta (volatility ratio) allowed"
            )
            
            st.session_state.settings['max_ltv'] = st.slider(
                "Max LTV Ratio", 0.50, 0.90, st.session_state.settings['max_ltv'], 0.05,
                help="Maximum loan-to-value ratio for leveraged positions"
            )
            
            st.session_state.settings['max_drawdown'] = st.slider(
                "Max Drawdown Tolerance", 0.10, 0.50, st.session_state.settings['max_drawdown'], 0.05,
                help="Maximum acceptable portfolio drawdown"
            )
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.session_state.settings['target_apr_min'] = st.number_input(
                    "Min Target APR", 0.01, 0.50, st.session_state.settings['target_apr_min'], 0.01,
                    help="Minimum acceptable annual percentage return"
                )
            with col2b:
                st.session_state.settings['target_apr_max'] = st.number_input(
                    "Max Target APR", 0.10, 1.00, st.session_state.settings['target_apr_max'], 0.05,
                    help="Maximum target annual percentage return"
                )
        
        with col3:
            st.markdown("#### 💰 Position & Asset Settings")
            
            # Asset pair selection
            available_pairs = get_available_coin_pairs()
            pair_labels = [f"{strong}/{weak}" for strong, weak in available_pairs]
            current_pair = f"{st.session_state.settings['strong_coin']}/{st.session_state.settings['weak_coin']}"
            
            if current_pair in pair_labels:
                default_index = pair_labels.index(current_pair)
            else:
                default_index = 0
            
            selected_pair = st.selectbox(
                "Trading Pair",
                pair_labels,
                index=default_index,
                help="Select the strong/weak asset pair for analysis"
            )
            
            if selected_pair:
                strong, weak = selected_pair.split('/')
                st.session_state.settings['strong_coin'] = strong
                st.session_state.settings['weak_coin'] = weak
            
            # Position sizing
            position_method = st.radio(
                "Position Sizing Method",
                ["Fixed", "Volatility-based"],
                index=0 if st.session_state.settings['position_sizing_method'] == 'fixed' else 1,
                help="Choose how to size positions"
            )
            st.session_state.settings['position_sizing_method'] = position_method.lower().replace('-', '_')
            
            st.session_state.settings['base_capital'] = st.number_input(
                "Base Capital ($)", 1000, 1000000, st.session_state.settings['base_capital'], 1000,
                help="Starting capital for backtesting"
            )
            
            if st.session_state.settings['position_sizing_method'] == 'volatility_based':
                st.session_state.settings['risk_per_trade'] = st.slider(
                    "Risk per Trade (%)", 0.01, 0.10, st.session_state.settings['risk_per_trade'], 0.005,
                    help="Percentage of capital to risk per trade"
                )
        
        # Settings summary and save/load
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset to Defaults", help="Reset all settings to default values"):
                # Reset settings to defaults
                st.session_state.settings = {
                    'entry_ps_threshold': 0.05,
                    'entry_as_threshold': 0.20,
                    'exit_as_threshold': 0.50,
                    'z_entry_threshold': -2.0,
                    'z_exit_threshold': 0.5,
                    'lookback_window': 120,
                    'mad_window': 120,
                    'correlation_threshold': 0.5,
                    'beta_min': 1.0,
                    'beta_max': 2.0,
                    'max_ltv': 0.75,
                    'target_apr_min': 0.08,
                    'target_apr_max': 0.25,
                    'max_drawdown': 0.30,
                    'strong_coin': 'ETH',
                    'weak_coin': 'BCH',
                    'use_z_score_signals': False,
                    'position_sizing_method': 'fixed',
                    'base_capital': 10000,
                    'risk_per_trade': 0.02
                }
                st.success("Settings reset to defaults!")
                st.rerun()
        
        with col2:
            if st.button("💾 Save Settings Profile", help="Save current settings as a preset"):
                st.info("Settings profile saved! (Feature coming soon)")
        
        with col3:
            if st.button("📁 Load Settings Profile", help="Load a saved settings preset"):
                st.info("Load settings profile! (Feature coming soon)")
        
        # Display current settings summary
        st.markdown("#### 📋 Current Settings Summary")
        settings_summary = f"""
        **Signal Type:** {'Z-score' if st.session_state.settings['use_z_score_signals'] else 'Percentile'}  
        **Trading Pair:** {st.session_state.settings['strong_coin']}/{st.session_state.settings['weak_coin']}  
        **Lookback Window:** {st.session_state.settings['lookback_window']} days  
        **Risk Filters:** Corr ≥ {st.session_state.settings['correlation_threshold']:.2f}, Beta ∈ [{st.session_state.settings['beta_min']:.1f}, {st.session_state.settings['beta_max']:.1f}]  
        **Position Sizing:** {st.session_state.settings['position_sizing_method'].replace('_', ' ').title()}  
        **Base Capital:** ${st.session_state.settings['base_capital']:,}
        """
        st.markdown(settings_summary)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab9:
        with st.expander("📖 Guide: Understanding Backtesting", expanded=False):
            st.markdown("""
            <div class="guide-content">
            This tab runs comprehensive backtests using your custom settings to evaluate strategy performance.

            <b>Backtest Features:</b>
            <ul>
                <li><b>Historical Simulation:</b> Applies your entry/exit rules to historical data to see how the strategy would have performed.</li>
                <li><b>Performance Metrics:</b> Calculates Sharpe ratio, win rate, maximum drawdown, and other key metrics.</li>
                <li><b>Benchmark Comparison:</b> Compares strategy returns against buy-and-hold ETH performance.</li>
                <li><b>Trade Analysis:</b> Shows individual trade details including duration and returns.</li>
            </ul>

            <b>Key Metrics Explained:</b>
            <ul>
                <li><b>Total Return:</b> Overall percentage gain/loss from the strategy.</li>
                <li><b>Sharpe Ratio:</b> Risk-adjusted return measure (higher is better, >1.0 is good).</li>
                <li><b>Max Drawdown:</b> Largest peak-to-trough decline (lower is better).</li>
                <li><b>Win Rate:</b> Percentage of profitable trades.</li>
                <li><b>Excess Return:</b> Strategy return minus buy-and-hold benchmark.</li>
            </ul>

            <b>How to Use:</b>
            <ul>
                <li><b>Adjust Settings:</b> Use the Settings tab to modify parameters.</li>
                <li><b>Run Backtest:</b> Click the backtest button to simulate historical performance.</li>
                <li><b>Analyze Results:</b> Review the performance charts and trade details.</li>
                <li><b>Optimize:</b> Iterate on settings to improve performance metrics.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ffffff !important; font-weight: 600;'>📊 Strategy Backtesting & Performance Analysis</h3>", unsafe_allow_html=True)
        
        # Backtest controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_period = st.selectbox(
                "Backtest Period",
                ["Full Dataset", "Last 6 Months", "Last 3 Months", "Last Month"],
                help="Select the time period for backtesting"
            )
        
        with col2:
            initial_capital = st.number_input(
                "Initial Capital ($)", 1000, 1000000, st.session_state.settings['base_capital'], 1000,
                help="Starting capital for the backtest"
            )
        
        with col3:
            if st.button("🚀 Run Backtest", help="Execute backtest with current settings"):
                with st.spinner("Running backtest..."):
                    # Filter data based on selected period
                    if backtest_period == "Last Month":
                        cutoff_date = filtered_df['ISO Date'].max() - pd.Timedelta(days=30)
                        backtest_df = filtered_df[filtered_df['ISO Date'] >= cutoff_date].copy()
                    elif backtest_period == "Last 3 Months":
                        cutoff_date = filtered_df['ISO Date'].max() - pd.Timedelta(days=90)
                        backtest_df = filtered_df[filtered_df['ISO Date'] >= cutoff_date].copy()
                    elif backtest_period == "Last 6 Months":
                        cutoff_date = filtered_df['ISO Date'].max() - pd.Timedelta(days=180)
                        backtest_df = filtered_df[filtered_df['ISO Date'] >= cutoff_date].copy()
                    else:
                        backtest_df = filtered_df.copy()
                    
                    # Ensure we have Z_score column for backtesting
                    if 'Z_score' not in backtest_df.columns:
                        backtest_df = calculate_strategy_metrics(backtest_df, st.session_state.settings['lookback_window'])
                    
                    # Run the backtest
                    try:
                        bt_results, trades, performance = run_backtest(backtest_df, st.session_state.settings, initial_capital)
                        
                        # Store results in session state
                        st.session_state.backtest_results = bt_results
                        st.session_state.backtest_trades = trades
                        st.session_state.backtest_performance = performance
                        
                        st.success(f"✅ Backtest completed! Found {len(trades)} trades over {len(backtest_df)} days.")
                        
                    except Exception as e:
                        st.error(f"❌ Backtest failed: {str(e)}")
        
        # Display backtest results if available
        if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results is not None:
            bt_results = st.session_state.backtest_results
            trades = st.session_state.backtest_trades
            performance = st.session_state.backtest_performance
            
            # Performance metrics cards
            st.markdown("#### 📊 Performance Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_return = performance['total_return']
                return_color = "positive" if total_return > 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {return_color}">{total_return:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                excess_return = performance['excess_return']
                excess_color = "positive" if excess_return > 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">vs ETH</div>
                    <div class="metric-value {excess_color}">{excess_return:+.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                sharpe = performance['sharpe_ratio']
                sharpe_color = "positive" if sharpe > 1.0 else "negative" if sharpe < 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {sharpe_color}">{sharpe:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                max_dd = performance['max_drawdown']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{max_dd:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                win_rate = performance['win_rate'] * 100
                win_color = "positive" if win_rate > 50 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value {win_color}">{win_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance chart
            st.markdown("#### 📈 Cumulative Performance")
            
            fig_performance = go.Figure()
            
            fig_performance.add_trace(go.Scatter(
                x=bt_results['ISO Date'],
                y=bt_results['strategy_return'],
                name='Strategy',
                line=dict(color='#22c55e', width=2)
            ))
            
            fig_performance.add_trace(go.Scatter(
                x=bt_results['ISO Date'],
                y=bt_results['eth_return'],
                name='Buy & Hold ETH',
                line=dict(color='#3b82f6', width=2)
            ))
            
            fig_performance.update_layout(
                title="Strategy vs Buy & Hold Performance",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=500,
                template="plotly_dark",
                hovermode='x unified',
                xaxis=dict(type="date", tickformat="%Y-%m-%d")
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Trade analysis
            if trades:
                st.markdown("#### 📋 Trade Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Trade Statistics:**")
                    st.write(f"• **Total Trades:** {performance['num_trades']}")
                    st.write(f"• **Win Rate:** {performance['win_rate']:.1%}")
                    st.write(f"• **Avg Trade Return:** {performance['avg_trade_return']:.2%}")
                    st.write(f"• **Avg Trade Duration:** {performance['avg_trade_duration']:.1f} days")
                
                with col2:
                    # Trade returns distribution
                    trade_returns = [t['return'] * 100 for t in trades]
                    fig_dist = go.Figure(data=[go.Histogram(x=trade_returns, nbinsx=20)])
                    fig_dist.update_layout(
                        title="Trade Returns Distribution",
                        xaxis_title="Return (%)",
                        yaxis_title="Frequency",
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Recent trades table
                st.markdown("**🔍 Recent Trades:**")
                trades_df = pd.DataFrame(trades)
                if len(trades_df) > 0:
                    trades_df['return_pct'] = trades_df['return'] * 100
                    display_trades = trades_df[['entry_date', 'exit_date', 'return_pct', 'duration_days']].tail(10)
                    display_trades.columns = ['Entry Date', 'Exit Date', 'Return (%)', 'Duration (days)']
                    st.dataframe(display_trades, use_container_width=True)
            
            # Settings used for this backtest
            st.markdown("#### ⚙️ Backtest Settings")
            settings_used = f"""
            **Signal Type:** {'Z-score' if st.session_state.settings['use_z_score_signals'] else 'Percentile'}  
            **Entry Threshold:** {st.session_state.settings['z_entry_threshold'] if st.session_state.settings['use_z_score_signals'] else f"PS: {st.session_state.settings['entry_ps_threshold']:.2%}, AS: {st.session_state.settings['entry_as_threshold']:.2%}"}  
            **Exit Threshold:** {st.session_state.settings['z_exit_threshold'] if st.session_state.settings['use_z_score_signals'] else f"AS: {st.session_state.settings['exit_as_threshold']:.2%}"}  
            **Lookback Window:** {st.session_state.settings['lookback_window']} days  
            **Risk Filters:** Corr ≥ {st.session_state.settings['correlation_threshold']:.2f}, Beta ∈ [{st.session_state.settings['beta_min']:.1f}, {st.session_state.settings['beta_max']:.1f}]
            """
            st.markdown(settings_used)
        
        else:
            st.info("👆 Click 'Run Backtest' to analyze strategy performance with your current settings.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Strategy summary
    st.header("🎯 Strategy Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="color: #ffffff !important;">
        <h3 style="color: #ffffff !important;">📋 Current Strategy Status</h3>
        <ul style="color: #ffffff !important;">
            <li style="color: #ffffff !important;"><b>Pair:</b> ETH (Strong) vs BCH (Weak)</li>
            <li style="color: #ffffff !important;"><b>Entry Logic:</b> PS %ile < 5% AND AS %ile < 20%</li>
            <li style="color: #ffffff !important;"><b>Exit Logic:</b> AS %ile > 50% OR Time Limit</li>
            <li style="color: #ffffff !important;"><b>Risk Controls:</b> Correlation ≥ 0.5, Beta ∈ [1,2]</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Latest values
        latest = filtered_df.iloc[-1]
        current_status = "🟢 ENTRY READY" if latest['entry_signal'] == 1 else "🔴 NO SIGNAL"
        if latest['exit_signal'] == 1:
            current_status = "🟡 EXIT SIGNAL"
        
        st.markdown(f"""
        <div style="color: #ffffff !important;">
        <h3 style="color: #ffffff !important;">📊 Latest Reading</h3>
        <ul style="color: #ffffff !important;">
            <li style="color: #ffffff !important;"><b>Status:</b> {current_status}</li>
            <li style="color: #ffffff !important;"><b>Point Spread:</b> {latest['Point Spread']:.4f}</li>
            <li style="color: #ffffff !important;"><b>PS Percentile:</b> {latest['PS_pct']:.1%}</li>
            <li style="color: #ffffff !important;"><b>AS Percentile:</b> {latest['AS_pct']:.1%}</li>
            <li style="color: #ffffff !important;"><b>Correlation:</b> {latest['Rolling Corr']:.3f}</li>
            <li style="color: #ffffff !important;"><b>Beta:</b> {latest['Rolling Beta']:.3f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Raw data section
    with st.expander("📋 View Raw Data"):
        st.dataframe(filtered_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='color: #ffffff !important; font-style: italic; text-align: center;'>Dashboard built for Ricardo's Pair Trading Strategy - Real-time analysis of statistical arbitrage opportunities</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()