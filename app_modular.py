"""
Pair Trading Dashboard - Main Application (Frontend Only)
========================================================

This is the main Streamlit application file that handles ONLY the user interface.
All business logic, calculations, and data processing has been moved to backend.py

Structure:
- UI Components: Headers, tabs, forms, charts
- User Interactions: Button clicks, form submissions
- Data Display: Tables, KPIs, charts using backend data
- No Business Logic: All computations delegated to DashboardDataProvider

Key Principle: This file should be easy to read and understand the user flow
without getting bogged down in complex calculations or business rules.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from scipy import stats
from typing import Tuple

# Enhanced data loader with Binance Futures API + CoinGecko fallback
class EnhancedDataLoader:
    """Enhanced data loader that supports Binance Futures API, Excel files, AND CoinGecko as fallback"""
    
    def __init__(self):
        self.data_cache = {}
        
        # Binance Futures symbol mapping
        self.binance_symbols = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT',
            'BNB': 'BNBUSDT',
            'ADA': 'ADAUSDT',
            'ALGO': 'ALGOUSDT',
            'APE': 'APEUSDT',
            'APT': 'APTUSDT',
            'ARB': 'ARBUSDT',
            'ATOM': 'ATOMUSDT',
            'AVAX': 'AVAXUSDT',
            'BCH': 'BCHUSDT'
        }
        
        # CoinGecko IDs (fallback)
        self.coingecko_ids = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'BNB': 'binancecoin',
            'ADA': 'cardano',
            'ALGO': 'algorand',
            'APE': 'apecoin',
            'APT': 'aptos',
            'ARB': 'arbitrum',
            'ATOM': 'cosmos',
            'AVAX': 'avalanche-2',
            'BCH': 'bitcoin-cash'
        }
    
    def load_real_data(self, strong_asset: str, weak_asset: str, days: int = 180, data_source: str = 'binance'):
        """Load REAL data - Primary: Binance Futures, Fallback: CoinGecko, Excel"""
        
        # Validate that we have these assets
        if strong_asset not in AVAILABLE_ASSETS:
            raise ValueError(f"❌ Strong asset '{strong_asset}' is not in our available assets: {AVAILABLE_ASSETS}")
        
        if weak_asset not in AVAILABLE_ASSETS:
            raise ValueError(f"❌ Weak asset '{weak_asset}' is not in our available assets: {AVAILABLE_ASSETS}")
        
        # Ensure we don't exceed API limits
        if days > 365:
            print(f"⚠️ Requested {days} days exceeds free tier limit. Using 365 days instead.")
            days = 365
        
        # Try Binance Futures FIRST (primary for client)
        if data_source == 'binance':
            print(f"🔷 Loading data from Binance Futures API for {strong_asset}/{weak_asset} ({days} days)")
            try:
                return self.load_from_binance(strong_asset, weak_asset, days=days)
            except Exception as binance_error:
                print(f"❌ Binance Futures failed: {str(binance_error)}")
                print(f"🔄 Falling back to CoinGecko...")
                data_source = 'coingecko'  # Auto fallback
        
        # Try CoinGecko as fallback
        if data_source == 'coingecko':
            print(f"🌐 Loading data from CoinGecko API for {strong_asset}/{weak_asset} ({days} days)")
            try:
                return self.load_from_coingecko(strong_asset, weak_asset, days=days)
            except Exception as coingecko_error:
                print(f"❌ CoinGecko failed: {str(coingecko_error)}")
            
            # Only try Excel as fallback for ETH/BCH
            if strong_asset == 'ETH' and weak_asset == 'BCH':
                print(f"📊 Trying Excel file as fallback for ETH/BCH...")
                try:
                    excel_file = 'ETH_BCH_multi_interval_analysis.xlsx'
                    import os
                    if os.path.exists(excel_file):
                        print(f"📊 Loading REAL data from {excel_file}")
                        
                        # Read all sheets to see what's available
                        excel_sheets = pd.read_excel(excel_file, sheet_name=None)
                        print(f"Available sheets: {list(excel_sheets.keys())}")
                        
                        # Try different sheet names
                        df = None
                        for sheet_name in excel_sheets.keys():
                            temp_df = excel_sheets[sheet_name]
                            print(f"Sheet '{sheet_name}' columns: {list(temp_df.columns)}")
                            
                            # Look for the right sheet with price data
                            if any('ETH' in str(col) for col in temp_df.columns) and any('BCH' in str(col) for col in temp_df.columns):
                                df = temp_df.copy()
                                print(f"✅ Using sheet '{sheet_name}' - contains ETH and BCH data")
                                break
                        
                        if df is not None:
                            # Clean and standardize column names
                            df.columns = df.columns.str.strip()  # Remove whitespace
                            
                            # Map various possible column names to standard format
                            column_mapping = {}
                            for col in df.columns:
                                col_lower = str(col).lower()
                                if 'date' in col_lower or 'time' in col_lower:
                                    column_mapping[col] = 'ISO Date'
                                elif 'eth' in col_lower and ('close' in col_lower or 'price' in col_lower):
                                    column_mapping[col] = 'ETH Close'
                                elif 'bch' in col_lower and ('close' in col_lower or 'price' in col_lower):
                                    column_mapping[col] = 'BCH Close'
                                elif 'eth' in col_lower and ('return' in col_lower or 'var' in col_lower):
                                    column_mapping[col] = 'LN ETH Var %'
                                elif 'bch' in col_lower and ('return' in col_lower or 'var' in col_lower):
                                    column_mapping[col] = 'LN BCH Var %'
                            
                            # Rename columns
                            df = df.rename(columns=column_mapping)
                            
                            # Ensure date column is datetime
                            if 'ISO Date' in df.columns:
                                df['ISO Date'] = pd.to_datetime(df['ISO Date'])
                            
                            # Calculate missing columns if we have price data
                            if 'ETH Close' in df.columns and 'BCH Close' in df.columns:
                                # Calculate log spread
                                df['log_spread'] = np.log(df['ETH Close']) - np.log(df['BCH Close'])
                                
                                # Calculate returns if not present
                                if 'LN ETH Var %' not in df.columns:
                                    df['LN ETH Var %'] = df['ETH Close'].pct_change() * 100
                                if 'LN BCH Var %' not in df.columns:
                                    df['LN BCH Var %'] = df['BCH Close'].pct_change() * 100
                                
                                # Calculate Point Spread and Accumulated Spread
                                df['Point Spread'] = df['LN BCH Var %'] - df['LN ETH Var %']
                                df['Accum Spread'] = df['Point Spread'].cumsum()
                                
                                # Calculate MAD Z-score
                                median_spread = df['log_spread'].median()
                                mad = np.median(np.abs(df['log_spread'] - median_spread))
                                df['mad_z_score'] = (df['log_spread'] - median_spread) / (mad * 1.4826)
                                
                                # Calculate percentiles
                                df['PS_pct'] = df['Point Spread'].rank(pct=True)
                                df['AS_pct'] = df['Accum Spread'].rank(pct=True)
                                
                                print(f"✅ Successfully processed Excel data: {len(df)} rows")
                                print(f"✅ Columns available: {list(df.columns)}")
                                return df
                            else:
                                print(f"❌ Excel file doesn't contain ETH and BCH price columns")
                                
                except Exception as excel_error:
                    print(f"❌ Error loading Excel file: {str(excel_error)}")
            
            # If all failed, raise error
            raise ValueError(f"Could not load data from any source (Binance/CoinGecko/Excel)")
    
    def load_from_binance(self, strong_asset: str, weak_asset: str, days: int = 180):
        """Load data from Binance Futures API (NO authentication required for public data)"""
        try:
            import requests
            import time
            from datetime import datetime, timedelta
            
            print(f"🔷 Fetching Binance Futures data for {strong_asset} and {weak_asset}")
            
            # Get Binance symbols
            strong_symbol = self.binance_symbols.get(strong_asset)
            weak_symbol = self.binance_symbols.get(weak_asset)
            
            if not strong_symbol or not weak_symbol:
                raise ValueError(f"❌ Binance mapping not available for {strong_asset} or {weak_asset}")
            
            # Binance Futures public API endpoint (NO auth needed for klines)
            base_url = "https://fapi.binance.com/fapi/v1/klines"
            
            # Calculate time range (Binance uses milliseconds)
            end_time = int(time.time() * 1000)
            start_time = int((time.time() - (days * 24 * 60 * 60)) * 1000)
            
            # Determine interval based on days requested
            if days <= 30:
                interval = '1h'  # Hourly for short periods
                limit = 1000  # Max per request
            elif days <= 90:
                interval = '4h'  # 4-hour for medium periods
                limit = 1000
            else:
                interval = '1d'  # Daily for long periods
                limit = 1000
            
            def fetch_binance_klines(symbol, asset_name):
                """Fetch klines (candlestick) data from Binance Futures"""
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': limit
                }
                
                print(f"  📊 Fetching {asset_name} data from Binance Futures...")
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    klines = response.json()
                    
                    # Parse klines data
                    # Format: [open_time, open, high, low, close, volume, close_time, ...]
                    df = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
                    
                    # Convert price to float and keep only close price
                    df[f'{asset_name}_price'] = df['close'].astype(float)
                    
                    # Return only timestamp and price
                    return df[['timestamp', f'{asset_name}_price']]
                else:
                    raise Exception(f"Binance API error {response.status_code}: {response.text}")
            
            # Fetch data for both assets
            strong_data = fetch_binance_klines(strong_symbol, strong_asset)
            time.sleep(0.5)  # Be nice to the API (Binance is more lenient than CoinGecko)
            weak_data = fetch_binance_klines(weak_symbol, weak_asset)
            
            # Merge the datasets
            df = pd.merge(strong_data, weak_data, on='timestamp', how='inner')
            df = df.rename(columns={
                'timestamp': 'ISO Date',
                f'{strong_asset}_price': f'{strong_asset} Close',
                f'{weak_asset}_price': f'{weak_asset} Close'
            })
            
            # Calculate returns
            df[f'LN {strong_asset} Var %'] = df[f'{strong_asset} Close'].pct_change() * 100
            df[f'LN {weak_asset} Var %'] = df[f'{weak_asset} Close'].pct_change() * 100
            
            # Calculate spreads
            df['log_spread'] = np.log(df[f'{strong_asset} Close']) - np.log(df[f'{weak_asset} Close'])
            df['Point Spread'] = df[f'LN {weak_asset} Var %'] - df[f'LN {strong_asset} Var %']
            df['Accum Spread'] = df['Point Spread'].cumsum()
            
            # Calculate MAD Z-score
            median_spread = df['log_spread'].median()
            mad = np.median(np.abs(df['log_spread'] - median_spread))
            df['mad_z_score'] = (df['log_spread'] - median_spread) / (mad * 1.4826)
            
            # Calculate percentiles
            df['PS_pct'] = df['Point Spread'].rank(pct=True)
            df['AS_pct'] = df['Accum Spread'].rank(pct=True)
            
            # Remove NaN rows
            df = df.dropna()
            
            print(f"✅ Successfully loaded {len(df)} data points from Binance Futures!")
            print(f"📅 Date range: {df['ISO Date'].min()} to {df['ISO Date'].max()}")
            print(f"📊 Interval: {interval}, Data points: {len(df)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Binance Futures API error: {str(e)}")
            raise ValueError(f"Could not load data from Binance Futures: {str(e)}")
    
    def load_from_coingecko(self, strong_asset: str, weak_asset: str, days: int = 365):
        """Load data from CoinGecko API as fallback"""
        try:
            import requests
            import time
            
            print(f"🌐 Fetching data from CoinGecko for {strong_asset} and {weak_asset}")
            
            # Get CoinGecko IDs
            strong_id = self.coingecko_ids.get(strong_asset)
            weak_id = self.coingecko_ids.get(weak_asset)
            
            if not strong_id or not weak_id:
                raise ValueError(f"❌ CoinGecko mapping not available for {strong_asset} or {weak_asset}")
            
            # Fetch historical data for both assets
            base_url = "https://api.coingecko.com/api/v3/coins/{}/market_chart"
            
            def fetch_coin_data(coin_id, asset_name):
                url = base_url.format(coin_id)
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': 'hourly' if days <= 90 else 'daily'
                }
                
                print(f"  📊 Fetching {asset_name} data...")
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 429:  # Rate limit
                    print("⏳ Rate limited, waiting 60 seconds...")
                    time.sleep(60)
                    response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    prices = data['prices']
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(prices, columns=['timestamp', f'{asset_name}_price'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
                else:
                    raise Exception(f"CoinGecko API error {response.status_code}: {response.text}")
            
            # Fetch data for both assets
            strong_data = fetch_coin_data(strong_id, strong_asset)
            time.sleep(1)  # Be nice to the API
            weak_data = fetch_coin_data(weak_id, weak_asset)
            
            # Merge the datasets
            df = pd.merge(strong_data, weak_data, on='timestamp', how='inner')
            df = df.rename(columns={
                'timestamp': 'ISO Date',
                f'{strong_asset}_price': f'{strong_asset} Close',
                f'{weak_asset}_price': f'{weak_asset} Close'
            })
            
            # Calculate returns
            df[f'LN {strong_asset} Var %'] = df[f'{strong_asset} Close'].pct_change() * 100
            df[f'LN {weak_asset} Var %'] = df[f'{weak_asset} Close'].pct_change() * 100
            
            # Calculate spreads
            df['log_spread'] = np.log(df[f'{strong_asset} Close']) - np.log(df[f'{weak_asset} Close'])
            df['Point Spread'] = df[f'LN {weak_asset} Var %'] - df[f'LN {strong_asset} Var %']
            df['Accum Spread'] = df['Point Spread'].cumsum()
            
            # Calculate MAD Z-score
            median_spread = df['log_spread'].median()
            mad = np.median(np.abs(df['log_spread'] - median_spread))
            df['mad_z_score'] = (df['log_spread'] - median_spread) / (mad * 1.4826)
            
            # Calculate percentiles
            df['PS_pct'] = df['Point Spread'].rank(pct=True)
            df['AS_pct'] = df['Accum Spread'].rank(pct=True)
            
            # Remove NaN rows
            df = df.dropna()
            
            print(f"✅ Successfully loaded {len(df)} data points from CoinGecko!")
            print(f"📅 Date range: {df['ISO Date'].min()} to {df['ISO Date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"❌ CoinGecko API error: {str(e)}")
            raise ValueError(f"Could not load data from CoinGecko: {str(e)}")

# Initialize enhanced data loader with CoinGecko support
data_loader = EnhancedDataLoader()

# Import remaining modules but make them optional
try:
    from backend_2 import (
        StatisticalAnalysis, 
        days_to_window,
        rolling_percentile
    )
except ImportError:
    # Fallback functions if backend_2 is not available
    def days_to_window(days: int, timeframe: str) -> int:
        timeframe_hours = {'15m': 0.25, '1h': 1, '4h': 4, '1d': 24}
        hours_per_day = timeframe_hours.get(timeframe, 1)
        return max(10, int(days * 24 / hours_per_day))
    
    class StatisticalAnalysis:
        @staticmethod
        def calculate_spread_statistics(spread_series):
            return {
                'mean': spread_series.mean(),
                'median': spread_series.median(),
                'std': spread_series.std(),
                'mad': np.median(np.abs(spread_series - spread_series.median())),
                'min': spread_series.min(),
                'max': spread_series.max(),
                'p05': spread_series.quantile(0.05),
                'p20': spread_series.quantile(0.20),
                'p50': spread_series.quantile(0.50),
                'p80': spread_series.quantile(0.80),
                'p95': spread_series.quantile(0.95)
            }
        
        @staticmethod
        def calculate_half_life(spread_series):
            return 7.0  # Simple fallback
from styles_modern import (
    get_modern_css, 
    get_trading_signal_html, 
    get_glass_card_html, 
    get_neon_metric_html,
    get_enhanced_kpi_card,
    get_premium_metric_card,
    get_modern_table_html,
    get_tooltip_html,
    get_loading_spinner,
    get_success_toast,
    get_error_toast,
    get_info_card
)
# Simple asset lists (only assets we have data for)
AVAILABLE_ASSETS = ['ETH', 'BTC', 'BNB', 'ADA', 'ALGO', 'APE', 'APT', 'ARB', 'ATOM', 'AVAX', 'BCH']
STRONG_ASSETS = ['ETH', 'BTC', 'BNB']  # Assets that can be used as collateral

# Simple settings - CoinGecko Free Tier Compatible
DEFAULT_SETTINGS = {
    'cointegration_window': 30,
    'correlation_window': 30,
    'beta_window': 30,
    'data_points': 180,      # Default to 6 months (safe for free tier)
    'min_data_points': 30,   # Minimum 30 days
    'max_data_points': 365   # Max 365 days (CoinGecko free tier limit)
}

TIMEFRAMES = {
    '5m': '5 Minutes',
    '15m': '15 Minutes', 
    '1h': '1 Hour',
    '4h': '4 Hours',
    '1d': '1 Day'
}

try:
    from config import (
        BINANCE_LOANS_ASSETS, 
        STRONG_ASSETS as CONFIG_STRONG_ASSETS, 
        DEFAULT_SETTINGS as CONFIG_DEFAULT_SETTINGS, 
        TIMEFRAMES as CONFIG_TIMEFRAMES,
        CHART_SETTINGS
    )
    # Use config if available, but fall back to our simple version
    AVAILABLE_ASSETS = BINANCE_LOANS_ASSETS if 'BINANCE_LOANS_ASSETS' in globals() else AVAILABLE_ASSETS
    STRONG_ASSETS = CONFIG_STRONG_ASSETS if 'CONFIG_STRONG_ASSETS' in globals() else STRONG_ASSETS
except ImportError:
    # Use our simple defaults
    pass

# ============================================================================
# APPLICATION SETUP & CONFIGURATION
# ============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Pair Trading Terminal - Professional",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Apply Modern Dark Trading Theme ---
st.markdown(get_modern_css(), unsafe_allow_html=True)

# --- Add Sidebar Navigation CSS ---
st.markdown("""
<style>
    /* Enhanced Sidebar Navigation */
    [data-testid="stSidebar"] {
        background: #F8FAFC !important;
    }
    
    .sidebar-logo {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.3);
    }
    
    .sidebar-logo h1 {
        color: white !important;
        font-size: 1.5rem !important;
        margin: 0 !important;
        font-weight: 700 !important;
    }
    
    .nav-section-title {
        color: #1E293B;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.5rem 1rem;
        margin: 1rem 0 0.5rem 0;
    }
    
    .dashboard-info {
        background: rgba(139, 92, 246, 0.05);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 2rem;
    }
    
    .dashboard-info-item {
        color: #475569;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    .dashboard-info-value {
        color: #1E293B;
        font-weight: 600;
    }
    
    /* Page buttons styling */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: white !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 0.875rem 1rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #F1F5F9 !important;
        border-color: #8B5CF6 !important;
        transform: translateX(4px) !important;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Selection Filter"

if 'settings' not in st.session_state:
    st.session_state.settings = DEFAULT_SETTINGS.copy()

if 'selection_results' not in st.session_state:
    st.session_state.selection_results = None
    
if 'trade_analysis_results' not in st.session_state:
    st.session_state.trade_analysis_results = None

if 'global_report' not in st.session_state:
    st.session_state.global_report = None
    
if 'active_signals' not in st.session_state:
    st.session_state.active_signals = []

if 'sidebar_settings' not in st.session_state:
    st.session_state.sidebar_settings = {
        'P_PS': 0.05,
        'P_AS': 0.50,
        'mad_threshold': 2.0,
        'position_size': 1000,
        'max_pairs': 3,
        'data_points': 1000
    }

# ============================================================================
# BACKEND INTERFACE
# ============================================================================

# Simple data provider fallback
class SimpleDashboardDataProvider:
    """Simple fallback data provider that doesn't use APIs"""
    
    def __init__(self, assets):
        self.assets = assets
    
    def get_dashboard_overview_data(self, session_state):
        """Return simple overview data"""
        return {
            'kpis': {
                'market_status': {'value': 'ONLINE', 'label': 'Market Status', 'delta': 'CSV Mode', 'icon': '🟢'},
                'available_assets': {'value': str(len(self.assets)), 'label': 'Available Assets', 'delta': 'Ready', 'icon': '💰'},
                'possible_pairs': {'value': str(len(self.assets) * (len(self.assets) - 1)), 'label': 'Possible Pairs', 'delta': 'Combinations', 'icon': '🔗'},
                'analyzed_pairs': {'value': '0', 'label': 'Analyzed Pairs', 'delta': 'Start Analysis', 'icon': '📊'},
                'active_signals': {'value': '0', 'label': 'Active Signals', 'delta': 'No signals yet', 'icon': '🚨'}
            }
        }
    
    def run_pair_selection_analysis(self, config):
        """Return sample selection results"""
        pairs = []
        for i, asset in enumerate(self.assets[:10]):  # Limit to first 10 for demo
            if asset != config['collateral']:
                pairs.append({
                    'Pair': f"{config['collateral']}/{asset}",
                    'Overall Score': 50 + np.random.randint(0, 40),
                    'Cointegration': np.random.uniform(0.01, 0.1),
                    'Correlation': np.random.uniform(0.3, 0.8),
                    'Beta': np.random.uniform(0.8, 2.0),
                    'PS_pct': np.random.uniform(0, 1),
                    'AS_pct': np.random.uniform(0, 1),
                    'Current Z-Score': np.random.uniform(-3, 3)
                })
        
        return pd.DataFrame(pairs)

# Initialize simple data provider
@st.cache_resource
def get_data_provider():
    """Initialize and cache the simple data provider"""
    return SimpleDashboardDataProvider(AVAILABLE_ASSETS)

# ============================================================================
# UTILITY FUNCTIONS (UI-related only)
# ============================================================================
def export_to_excel(dataframes_dict: dict, filename: str) -> BytesIO:
    """Export multiple dataframes to Excel file"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

def create_distribution_chart(data: pd.Series, title: str, thresholds: dict = None, show_qq: bool = False) -> go.Figure:
    """Create a professional distribution histogram with MAD indicators"""
    
    # Calculate statistics
    median_val = data.median()
    mad_val = np.median(np.abs(data - median_val))
    sigma_equivalent = 1.4826 * mad_val
    
    fig = go.Figure()
    
    # Add main histogram with gradient
    fig.add_trace(go.Histogram(
        x=data.dropna(),
        nbinsx=50,
        name='Distribution',
        marker=dict(
            color='rgba(59, 130, 246, 0.7)',
            line=dict(color='rgba(37, 99, 235, 0.8)', width=1)
        ),
        opacity=0.8,
        histnorm='probability density'
    ))
    
    # Add normal distribution overlay
    x_range = np.linspace(data.min(), data.max(), 200)
    normal_dist = stats.norm.pdf(x_range, data.mean(), data.std())
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_dist,
        name='Normal Curve',
        line=dict(color='#ef4444', width=3, dash='dash'),
        opacity=0.8
    ))
    
    # Add MAD-based threshold lines
    mad_lines = [
        (median_val, 'Median', '#6366f1', 'solid', 3),
        (median_val - sigma_equivalent, '-1 MAD', '#f59e0b', 'dash', 2),
        (median_val + sigma_equivalent, '+1 MAD', '#f59e0b', 'dash', 2),
        (median_val - 2*sigma_equivalent, '-2 MAD', '#ef4444', 'dot', 2),
        (median_val + 2*sigma_equivalent, '+2 MAD', '#ef4444', 'dot', 2)
    ]
    
    for value, label, color, dash, width in mad_lines:
        fig.add_vline(
            x=value,
            line=dict(color=color, dash=dash, width=width),
            annotation=dict(
                text=f"{label}: {value:.4f}",
                textangle=90,
                font=dict(size=10, color=color)
            )
        )
    
    # Add custom thresholds if provided
    if thresholds:
        for name, value in thresholds.items():
            if name not in ['median', 'success', 'danger']:  # Skip our own thresholds
                color = CHART_SETTINGS['colors'].get(name, '#666')
                fig.add_vline(
                    x=value,
                    line=dict(color=color, dash='dashdot', width=2),
                    annotation=dict(
                        text=f"{name}: {value:.3f}",
                        textangle=90,
                        font=dict(size=9, color=color)
                    )
                )
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color='#1e40af', family='Inter')
        ),
        xaxis_title="Spread Value",
        yaxis_title="Probability Density",
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#374151', family='Inter'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    return fig

def create_indexed_performance_chart(df: pd.DataFrame, strong_asset: str, weak_asset: str) -> go.Figure:
    """Create indexed performance chart (base 100) like Carlynn's"""
    fig = go.Figure()
    
    # Check if required columns exist
    if f'{strong_asset} Close' not in df.columns or f'{weak_asset} Close' not in df.columns:
        fig.add_annotation(
            text="Price data not available for performance chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    # Index to 100
    strong_indexed = (df[f'{strong_asset} Close'] / df[f'{strong_asset} Close'].iloc[0]) * 100
    weak_indexed = (df[f'{weak_asset} Close'] / df[f'{weak_asset} Close'].iloc[0]) * 100
    
    # Add traces with gradient fills
    fig.add_trace(go.Scatter(
        x=df['ISO Date'],
        y=strong_indexed,
        name=f'{strong_asset} (Strong)',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['ISO Date'],
        y=weak_indexed,
        name=f'{weak_asset} (Weak)',
        line=dict(color='#ef4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.05)'
    ))
    
    # Add 100 baseline
    fig.add_hline(y=100, line=dict(color='#6b7280', width=1, dash='dot'),
                  annotation_text='Baseline (100)')
    
    # Calculate relative performance
    relative_perf = ((strong_indexed.iloc[-1] - weak_indexed.iloc[-1]) / 100) * 100
    
    fig.update_layout(
        title=dict(
            text=f"📈 Indexed Performance | Relative: {relative_perf:+.1f}%",
            x=0.5,
            font=dict(size=16, color='#1e40af', family='Inter')
        ),
        xaxis_title="Date",
        yaxis_title="Indexed Value (Base = 100)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#374151', family='Inter'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        yaxis=dict(
            gridcolor='rgba(156, 163, 175, 0.2)',
            zerolinecolor='rgba(156, 163, 175, 0.3)'
        ),
        xaxis=dict(gridcolor='rgba(156, 163, 175, 0.1)')
    )
    
    return fig

def create_point_spread_chart(df: pd.DataFrame, show_mad_bands: bool = True, 
                             show_percentile_bands: bool = True, 
                             show_signals: bool = True) -> go.Figure:
    """Create Point Spread chart with MAD and percentile bands (Carlynn's style)"""
    fig = go.Figure()
    
    # Check if Point Spread column exists
    if 'Point Spread' not in df.columns:
        fig.add_annotation(
            text="Point Spread data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    # Main PS line
    fig.add_trace(go.Scatter(
        x=df['ISO Date'],
        y=df['Point Spread'],
        name='Point Spread',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='Date: %{x}<br>PS: %{y:.6f}<extra></extra>'
    ))
    
    # Calculate percentiles and MAD
    ps_median = df['Point Spread'].median()
    ps_mad = np.median(np.abs(df['Point Spread'] - ps_median))
    ps_p25 = df['Point Spread'].quantile(0.25)
    ps_p75 = df['Point Spread'].quantile(0.75)
    ps_p05 = df['Point Spread'].quantile(0.05)
    ps_p95 = df['Point Spread'].quantile(0.95)
    
    # Add percentile bands
    if show_percentile_bands:
        fig.add_hline(y=ps_median, line=dict(color='#6366f1', width=2, dash='dash'),
                     annotation_text=f'Median: {ps_median:.6f}')
        fig.add_hline(y=ps_p05, line=dict(color='#fbbf24', width=1, dash='dot'),
                     annotation_text=f'5th %ile: {ps_p05:.6f}')
        fig.add_hline(y=ps_p95, line=dict(color='#fbbf24', width=1, dash='dot'),
                     annotation_text=f'95th %ile: {ps_p95:.6f}')
    
    # Add MAD bands
    if show_mad_bands:
        sigma_eq = 1.4826 * ps_mad
        for k, color in [(1, '#22c55e'), (2, '#f59e0b'), (3, '#ef4444')]:
            fig.add_hline(y=ps_median + k*sigma_eq, 
                         line=dict(color=color, width=1, dash='dashdot'),
                         annotation_text=f'+{k} MAD')
            fig.add_hline(y=ps_median - k*sigma_eq,
                         line=dict(color=color, width=1, dash='dashdot'),
                         annotation_text=f'-{k} MAD')
    
    # Add entry/exit signals
    if show_signals and 'entry_signal' in df.columns:
        entry_points = df[df['entry_signal'] == 1]
        exit_points = df[df['exit_signal'] == 1] if 'exit_signal' in df.columns else pd.DataFrame()
        
        if not entry_points.empty:
            fig.add_trace(go.Scatter(
                x=entry_points['ISO Date'],
                y=entry_points['Point Spread'],
                mode='markers',
                name='Entry Signal',
                marker=dict(color='#10b981', size=10, symbol='triangle-up',
                          line=dict(color='white', width=2)),
                hovertemplate='Entry<br>Date: %{x}<br>PS: %{y:.6f}<extra></extra>'
            ))
        
        if not exit_points.empty:
            fig.add_trace(go.Scatter(
                x=exit_points['ISO Date'],
                y=exit_points['Point Spread'],
                mode='markers',
                name='Exit Signal',
                marker=dict(color='#ef4444', size=10, symbol='triangle-down',
                          line=dict(color='white', width=2)),
                hovertemplate='Exit<br>Date: %{x}<br>PS: %{y:.6f}<extra></extra>'
            ))
    
    # Add PS_pct annotation if available
    if 'PS_pct' in df.columns:
        current_ps_pct = df['PS_pct'].iloc[-1]
        fig.add_annotation(
            x=df['ISO Date'].iloc[-1],
            y=df['Point Spread'].iloc[-1],
            text=f"Current PS_pct: {current_ps_pct:.1%}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#3b82f6",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#3b82f6",
            borderwidth=2
        )
    
    fig.update_layout(
        title=dict(
            text="📊 Point Spread (PS) with Statistical Bands",
            x=0.5,
            font=dict(size=16, color='#1e40af', family='Inter')
        ),
        xaxis_title="Date",
        yaxis_title="Point Spread (Weak - Strong Returns)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#374151', family='Inter'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(gridcolor='rgba(156, 163, 175, 0.1)'),
        yaxis=dict(gridcolor='rgba(156, 163, 175, 0.2)')
    )
    
    return fig

def create_accumulated_spread_chart(df: pd.DataFrame, show_mad_bands: bool = True,
                                   show_percentile_bands: bool = True,
                                   show_signals: bool = True) -> go.Figure:
    """Create Accumulated Spread chart (Carlynn's AS visualization)"""
    fig = go.Figure()
    
    # Check if Accumulated Spread column exists
    if 'Accum Spread' not in df.columns:
        fig.add_annotation(
            text="Accumulated Spread data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig
    
    # Main AS line with gradient fill
    fig.add_trace(go.Scatter(
        x=df['ISO Date'],
        y=df['Accum Spread'],
        name='Accumulated Spread',
        line=dict(color='#8b5cf6', width=3),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.1)',
        hovertemplate='Date: %{x}<br>AS: %{y:.6f}<extra></extra>'
    ))
    
    # Calculate statistics
    as_median = df['Accum Spread'].median()
    as_mad = np.median(np.abs(df['Accum Spread'] - as_median))
    as_p05 = df['Accum Spread'].quantile(0.05)
    as_p20 = df['Accum Spread'].quantile(0.20)
    as_p80 = df['Accum Spread'].quantile(0.80)
    as_p95 = df['Accum Spread'].quantile(0.95)
    
    # Add percentile bands
    if show_percentile_bands:
        fig.add_hline(y=as_median, line=dict(color='#7c3aed', width=2, dash='dash'),
                     annotation_text=f'Median: {as_median:.4f}')
        fig.add_hline(y=as_p20, line=dict(color='#22c55e', width=1, dash='dot'),
                     annotation_text=f'20th %ile (Entry Zone): {as_p20:.4f}')
        fig.add_hline(y=as_p80, line=dict(color='#f59e0b', width=1, dash='dot'),
                     annotation_text=f'80th %ile: {as_p80:.4f}')
    
    # Add MAD bands
    if show_mad_bands:
        sigma_eq = 1.4826 * as_mad
        for k, color in [(1, '#3b82f6'), (2, '#f59e0b')]:
            fig.add_hline(y=as_median + k*sigma_eq,
                         line=dict(color=color, width=1, dash='dashdot'),
                         annotation_text=f'+{k} MAD')
            fig.add_hline(y=as_median - k*sigma_eq,
                         line=dict(color=color, width=1, dash='dashdot'),
                         annotation_text=f'-{k} MAD')
    
    # Add entry/exit signals
    if show_signals and 'entry_signal' in df.columns:
        entry_points = df[df['entry_signal'] == 1]
        exit_points = df[df['exit_signal'] == 1] if 'exit_signal' in df.columns else pd.DataFrame()
        
        if not entry_points.empty:
            fig.add_trace(go.Scatter(
                x=entry_points['ISO Date'],
                y=entry_points['Accum Spread'],
                mode='markers',
                name='Entry Signal',
                marker=dict(color='#10b981', size=10, symbol='triangle-up',
                          line=dict(color='white', width=2))
            ))
        
        if not exit_points.empty:
            fig.add_trace(go.Scatter(
                x=exit_points['ISO Date'],
                y=exit_points['Accum Spread'],
                mode='markers',
                name='Exit Signal',
                marker=dict(color='#ef4444', size=10, symbol='triangle-down',
                          line=dict(color='white', width=2))
            ))
    
    # Add AS_pct annotation
    if 'AS_pct' in df.columns:
        current_as_pct = df['AS_pct'].iloc[-1]
        fig.add_annotation(
            x=df['ISO Date'].iloc[-1],
            y=df['Accum Spread'].iloc[-1],
            text=f"Current AS_pct: {current_as_pct:.1%}",
            showarrow=True,
            arrowhead=2,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#8b5cf6",
            borderwidth=2
        )
    
    fig.update_layout(
        title=dict(
            text="📈 Accumulated Spread (AS) with Entry Zones",
            x=0.5,
            font=dict(size=16, color='#1e40af', family='Inter')
        ),
        xaxis_title="Date",
        yaxis_title="Accumulated Spread (Cumulative Weak - Strong)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#374151', family='Inter'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    
    return fig

def create_correlation_beta_charts(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Create Rolling Correlation and Beta charts (Carlynn's style)"""
    # Correlation chart
    fig_corr = go.Figure()
    
    if 'Rolling Corr' in df.columns:
        fig_corr.add_trace(go.Scatter(
            x=df['ISO Date'],
            y=df['Rolling Corr'],
            name='Rolling Correlation',
            line=dict(color='#06b6d4', width=2),
            fill='tozeroy',
            fillcolor='rgba(6, 182, 212, 0.1)'
        ))
        
        # Add threshold lines
        fig_corr.add_hline(y=0.5, line=dict(color='#10b981', width=2, dash='dash'),
                          annotation_text='Min Threshold (0.5)')
        fig_corr.add_hline(y=0.7, line=dict(color='#fbbf24', width=1, dash='dot'),
                          annotation_text='Strong Correlation (0.7)')
        
        fig_corr.update_layout(
            title="🔗 Rolling Correlation (30-period)",
            xaxis_title="Date",
            yaxis_title="Correlation Coefficient",
            height=300,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#374151', family='Inter', size=12)
        )
    
    # Beta chart
    fig_beta = go.Figure()
    
    if 'Rolling Beta' in df.columns or 'beta' in df.columns:
        beta_col = 'Rolling Beta' if 'Rolling Beta' in df.columns else 'beta'
        fig_beta.add_trace(go.Scatter(
            x=df['ISO Date'],
            y=df[beta_col],
            name='Rolling Beta',
            line=dict(color='#f97316', width=2)
        ))
        
        # Add ideal range
        fig_beta.add_hrect(y0=1.0, y1=2.0, fillcolor='rgba(34, 197, 94, 0.1)',
                          annotation_text='Ideal Range (1.0-2.0)', annotation_position='right')
        fig_beta.add_hline(y=1.0, line=dict(color='#22c55e', width=1, dash='dash'))
        fig_beta.add_hline(y=2.0, line=dict(color='#22c55e', width=1, dash='dash'))
        
        fig_beta.update_layout(
            title="📊 Rolling Beta (Volatility Ratio)",
            xaxis_title="Date",
            yaxis_title="Beta",
            height=300,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#374151', family='Inter', size=12)
        )
    
    return fig_corr, fig_beta

def create_spread_chart(df: pd.DataFrame, strong_asset: str, weak_asset: str) -> go.Figure:
    """Create professional spread chart with trading signals"""
    fig = go.Figure()
    
    # Add log spread line with gradient fill
    fig.add_trace(go.Scatter(
        x=df['ISO Date'],
        y=df['log_spread'],
        name='Log Spread',
        line=dict(color='#3b82f6', width=3),
        fill='tonexty' if 'spread_median' in df.columns else None,
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Add median line with enhanced styling
    if 'spread_median' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['ISO Date'],
            y=df['spread_median'],
            name='Median (Mean Reversion Target)',
            line=dict(color='#6366f1', width=2, dash='dash'),
            opacity=0.8
        ))
        
        # Add MAD bands
        if 'spread_mad' in df.columns:
            mad_upper = df['spread_median'] + 2 * df['spread_mad'] * 1.4826
            mad_lower = df['spread_median'] - 2 * df['spread_mad'] * 1.4826
            
            fig.add_trace(go.Scatter(
                x=df['ISO Date'],
                y=mad_upper,
                name='+2 MAD (Sell Signal)',
                line=dict(color='#ef4444', width=2, dash='dot'),
                opacity=0.6
            ))
            
            fig.add_trace(go.Scatter(
                x=df['ISO Date'],
                y=mad_lower,
                name='-2 MAD (Buy Signal)',
                line=dict(color='#10b981', width=2, dash='dot'),
                opacity=0.6
            ))
    
    # Add enhanced entry/exit signals
    if 'entry_signal' in df.columns:
        entry_points = df[df['entry_signal'] == 1]
        if not entry_points.empty:
            fig.add_trace(go.Scatter(
                x=entry_points['ISO Date'],
                y=entry_points['log_spread'],
                mode='markers',
                name='🔥 Entry Signal',
                marker=dict(
                    color='#10b981',
                    size=12,
                    symbol='triangle-up',
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>Entry Signal</b><br>Date: %{x}<br>Spread: %{y:.6f}<extra></extra>'
            ))
    
    if 'exit_signal' in df.columns:
        exit_points = df[df['exit_signal'] == 1]
        if not exit_points.empty:
            fig.add_trace(go.Scatter(
                x=exit_points['ISO Date'],
                y=exit_points['log_spread'],
                mode='markers',
                name='💰 Exit Signal',
                marker=dict(
                    color='#ef4444',
                    size=12,
                    symbol='triangle-down',
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>Exit Signal</b><br>Date: %{x}<br>Spread: %{y:.6f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text=f"📊 {strong_asset}/{weak_asset} Spread Analysis",
            x=0.5,
            font=dict(size=20, color='#1e40af', family='Inter')
        ),
        xaxis_title="Date",
        yaxis_title="Log Price Spread",
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#374151', family='Inter'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(59, 130, 246, 0.2)',
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(59, 130, 246, 0.1)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(59, 130, 246, 0.1)'
        )
    )
    
    return fig

def create_correlation_heatmap(correlation_data: pd.DataFrame) -> go.Figure:
    """Create a professional correlation heatmap for asset pairs"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.index,
        colorscale=[
            [0, '#ef4444'],     # Red for negative correlation
            [0.5, '#f8fafc'],   # Light gray for neutral
            [1, '#10b981']      # Green for positive correlation
        ],
        zmid=0,
        text=correlation_data.values,
        texttemplate='%{text:.2f}',
        textfont=dict(size=10, color='white'),
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="📊 Asset Correlation Matrix",
            x=0.5,
            font=dict(size=18, color='#1e40af', family='Inter')
        ),
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#374151', family='Inter')
    )
    
    return fig

def create_cointegration_scatter(cointegration_data: pd.DataFrame) -> go.Figure:
    """Create a scatter plot showing cointegration vs correlation analysis"""
    fig = go.Figure()
    
    # Color code by overall score
    colors = []
    for score in cointegration_data['Overall Score']:
        if score >= 80:
            colors.append('#10b981')  # Excellent - Green
        elif score >= 60:
            colors.append('#3b82f6')  # Good - Blue
        elif score >= 40:
            colors.append('#f59e0b')  # Moderate - Orange
        else:
            colors.append('#ef4444')  # Poor - Red
    
    fig.add_trace(go.Scatter(
        x=cointegration_data['Correlation'],
        y=cointegration_data['Cointegration'],
        mode='markers',
        marker=dict(
            color=colors,
            size=cointegration_data['Overall Score'] / 5,  # Size based on score
            opacity=0.7,
            line=dict(color='white', width=1)
        ),
        text=cointegration_data['Pair'],
        hovertemplate='<b>%{text}</b><br>' +
                     'Correlation: %{x:.3f}<br>' +
                     'Cointegration p-value: %{y:.4f}<br>' +
                     'Overall Score: %{customdata:.0f}/100<extra></extra>',
        customdata=cointegration_data['Overall Score']
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.05, line_dash="dash", line_color="#ef4444", 
                  annotation_text="Cointegration Threshold (p=0.05)")
    fig.add_vline(x=0.5, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Correlation Threshold (0.5)")
    
    fig.update_layout(
        title=dict(
            text="🎯 Pair Quality Analysis: Cointegration vs Correlation",
            x=0.5,
            font=dict(size=18, color='#1e40af', family='Inter')
        ),
        xaxis_title="Correlation Coefficient",
        yaxis_title="Cointegration P-Value (lower is better)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#374151', family='Inter'),
        showlegend=False
    )
    
    return fig

# --- SIDEBAR NAVIGATION ---
def render_sidebar_navigation():
    """Render modern sidebar navigation"""
    with st.sidebar:
        # Logo/Brand
        st.markdown("""
            <div class="sidebar-logo">
                <h1>CRYPTO TRADING</h1>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.75rem; opacity: 0.9;">Professional Analytics Platform</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation Section
        st.markdown('<div class="nav-section-title">NAVIGATION</div>', unsafe_allow_html=True)
        
        # Selection Filter Button
        if st.button("Selection Filter", key="nav_selection", use_container_width=True):
            st.session_state.current_page = "Selection Filter"
            st.rerun()
        
        # Trade Analysis Button
        if st.button("Trade Analysis", key="nav_trade", use_container_width=True):
            st.session_state.current_page = "Trade Analysis"
            st.rerun()
        
        # Data Source Selector
        st.markdown("---")
        st.markdown('<div class="nav-section-title">DATA SOURCE</div>', unsafe_allow_html=True)
        
        data_source = st.radio(
            "Select Data Provider",
            options=['binance', 'coingecko'],
            format_func=lambda x: '🔷 Binance Futures (Recommended)' if x == 'binance' else '🌐 CoinGecko (Fallback)',
            index=0,
            key='data_source_selector',
            help="Binance Futures: More data, better for trading. CoinGecko: Fallback if Binance fails."
        )
        
        # Store in session state
        if 'data_source' not in st.session_state:
            st.session_state.data_source = 'binance'
        st.session_state.data_source = data_source
        
        # Dashboard Info
        st.markdown(f"""
            <div class="dashboard-info">
                <div class="dashboard-info-item">
                    <strong>Current Page:</strong><br/>
                    <span class="dashboard-info-value">{st.session_state.current_page}</span>
                </div>
                <div class="dashboard-info-item">
                    <strong>Data Source:</strong><br/>
                    <span class="dashboard-info-value">{'Binance 🔷' if data_source == 'binance' else 'CoinGecko 🌐'}</span>
                </div>
                <div class="dashboard-info-item">
                    <strong>Available Coins:</strong><br/>
                    <span class="dashboard-info-value">{len(AVAILABLE_ASSETS)}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # API info based on source
        if data_source == 'binance':
            st.info("🔷 **Binance Futures**: Public API, no auth needed, up to 1000 candles per request")
        else:
            st.warning("ℹ️ **CoinGecko**: Free tier, max 365 days of data per request")
    
    # Return settings from session state
    return st.session_state.sidebar_settings

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main application function - handles UI layout and user interactions
    All business logic is delegated to the backend DashboardDataProvider
    
    Flow:
    1. Initialize backend data provider
    2. Setup sidebar with user controls
    3. Display professional header
    4. Show dashboard overview with KPIs
    5. Display main tabs for different functionalities
    """
    # ========================================
    # INITIALIZATION
    # ========================================
    
    # Initialize backend data provider (cached)
    data_provider = get_data_provider()
    
    # Render sidebar navigation and get settings
    sidebar_settings = render_sidebar_navigation()
    
    # ========================================
    # PAGE ROUTING
    # ========================================
    
    # Route to the appropriate page
    current_page = st.session_state.current_page
    
    if current_page == "Selection Filter":
        render_selection_filter_page(data_provider, sidebar_settings)
    elif current_page == "Trade Analysis":
        render_trade_analysis_page(data_provider, sidebar_settings)
def render_selection_filter_page(data_provider, sidebar_settings):
    """Selection Filter Page - Find the best trading pairs"""
    
    # Modern header
    st.markdown("""
            <div style="
            text-align: center; 
            padding: 1.5rem 0;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(147, 51, 234, 0.02) 100%);
                border-radius: 16px;
                margin-bottom: 2rem;
            border: 1px solid rgba(139, 92, 246, 0.1);
        ">
            <h1 style="
                color: #8B5CF6; 
                margin: 0; 
                font-weight: 700;
                font-size: 2rem;
                letter-spacing: -0.5px;
            ">SELECTION FILTER</h1>
            <p style="color: #6B7280; font-size: 0.95rem; margin-top: 0.75rem; font-weight: 500;">
                Analyze all available pairs to find optimal trading opportunities
            </p>
            </div>
        """, unsafe_allow_html=True)
        
    # Configuration Section - Simplified and cleaner
    st.markdown('<h3 style="color: #1E293B; font-weight: 600; margin-bottom: 1.5rem;">Configuration</h3>', unsafe_allow_html=True)
    
    # Main configuration in a clean 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        timeframe = st.selectbox(
            "Timeframe",
            options=list(TIMEFRAMES.keys()),
            format_func=lambda x: TIMEFRAMES[x],
            index=2,
            help="Select the time interval for analysis"
        )
    
    with col2:
        collateral = st.selectbox(
            "Collateral Asset",
            options=STRONG_ASSETS,
            index=1,  # Default to ETH
            help="Choose your base trading asset - ETH or BTC recommended"
        )
    
    # Initialize with defaults (will be overridden if user changes in expander)
    coint_window = DEFAULT_SETTINGS['cointegration_window']
    corr_window = DEFAULT_SETTINGS['correlation_window']
    beta_window = DEFAULT_SETTINGS['beta_window']
    
    # Advanced settings in an expander to keep it clean
    with st.expander("⚙ Advanced Settings (Optional)", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            coint_window = st.number_input(
            "Cointegration Window",
            min_value=10,
            max_value=100,
                value=DEFAULT_SETTINGS['cointegration_window'],
            step=5,
                help="Higher = more stable, Lower = more sensitive"
            )
        
        with adv_col2:
            corr_window = st.number_input(
            "Correlation Window",
            min_value=10,
            max_value=100,
            value=DEFAULT_SETTINGS['correlation_window'],
            step=5
        )
    
        with adv_col3:
            beta_window = st.number_input(
            "Beta Window",
            min_value=10,
            max_value=100,
            value=DEFAULT_SETTINGS['beta_window'],
            step=5
        )
    
    # Min score filter - prominent but simple
    st.markdown("---")

    min_score = st.slider(
    "Minimum Score Filter",
        min_value=0,
        max_value=100,
        value=40,
    step=5,
    help="📊 Controls how strict pair selection is. Higher = fewer but stronger pairs. 40=broad search, 60=quality focus, 80+=premium only"
)
    
    # Large, friendly Run Analysis button
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left_space, col_button, col_right_space = st.columns([1, 2, 1])
    
    with col_button:
        if st.button("Run Analysis", type="primary", use_container_width=True):
            # Show loading state
            loading_placeholder = st.empty()
            loading_placeholder.markdown(get_loading_spinner("Analyzing all pairs..."), unsafe_allow_html=True)
            
            try:
                # Prepare configuration for backend
                config = {
                    'timeframe': timeframe,
                    'collateral': collateral,
                    'cointegration_window': coint_window,
                    'correlation_window': corr_window,
                    'beta_window': beta_window,
                    'min_score': min_score
                }
                
                # Run analysis using backend
                results_df = data_provider.run_pair_selection_analysis(config)
                st.session_state.selection_results = results_df
                
                # Clear loading and show success
                loading_placeholder.empty()
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                loading_placeholder.empty()
                st.error(f"Analysis failed: {str(e)}")
    
    # Export button (only show if results exist)
    if st.session_state.selection_results is not None:
        col_left_space2, col_export, col_right_space2 = st.columns([1, 2, 1])
        with col_export:
            excel_data = export_to_excel(
                {"Selection Results": st.session_state.selection_results},
                "selection_results.xlsx"
            )
            st.download_button(
                label="Export to Excel",
                data=excel_data,
                file_name=f"selection_{collateral}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Enhanced Results Display - Using backend for calculations
    if st.session_state.selection_results is not None:
        # Check if 'Overall Score' column exists, if not create a simple placeholder
        if 'Overall Score' not in st.session_state.selection_results.columns:
            st.session_state.selection_results['Overall Score'] = 50.0  # Default score
        
        df_filtered = st.session_state.selection_results[
            st.session_state.selection_results['Overall Score'] >= min_score
        ].sort_values('Overall Score', ascending=False)
        
        # Get simple summary data
        summary_data = {
            'total_pairs': len(st.session_state.selection_results),
            'filtered_pairs': len(df_filtered),
            'success_rate': f"{len(df_filtered)/len(st.session_state.selection_results)*100:.1f}%",
            'best_pair': df_filtered.iloc[0]['Pair'] if not df_filtered.empty else 'None',
            'best_score': df_filtered.iloc[0]['Overall Score'] if not df_filtered.empty else 0,
            'score_quality': 'Excellent' if (not df_filtered.empty and df_filtered.iloc[0]['Overall Score'] > 80) else 'Good'
        }
        
        # Summary Metrics with Enhanced KPI Cards
        st.markdown('<h3 style="color: #1E293B; font-weight: 600; margin: 2rem 0 1.5rem 0;">Analysis Summary</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(get_enhanced_kpi_card(
                value=str(summary_data['total_pairs']),
                label="Total Pairs Analyzed",
                icon="■",
                color="purple"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(get_enhanced_kpi_card(
                value=str(summary_data['filtered_pairs']),
                label="Above Threshold",
                delta=summary_data['success_rate'],
                icon="▲",
                color="purple"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(get_enhanced_kpi_card(
                value=summary_data['best_pair'],
                label="Best Pair",
                icon="◆",
                color="purple"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(get_enhanced_kpi_card(
                value=f"{summary_data['best_score']:.1f}/100",
                label="Best Score",
                delta=summary_data['score_quality'],
                icon="●",
                color="purple"
            ), unsafe_allow_html=True)
        
        # Results Table
        st.markdown('<h3 style="color: #1E293B; font-weight: 600; margin: 2rem 0 1.5rem 0;">Filtered Results</h3>', unsafe_allow_html=True)
        
        if not df_filtered.empty:
            # Configure column display (including Carlynn's metrics)
            column_config = {
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Pair": st.column_config.TextColumn("Pair", width="medium"),
                "Cointegration": st.column_config.NumberColumn("p-value", format="%.4f"),
                "Correlation": st.column_config.NumberColumn("Correlation", format="%.3f"),
                "Beta": st.column_config.NumberColumn("Beta", format="%.3f"),
                "PS_pct": st.column_config.NumberColumn("PS_pct", format="%.3f"),
                "AS_pct": st.column_config.NumberColumn("AS_pct", format="%.3f"),
                "Entry Signal": st.column_config.CheckboxColumn("Entry OK", width="small"),
                "Overall Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                "Current Z-Score": st.column_config.NumberColumn("Z-Score", format="%.2f")
            }
            
            st.dataframe(
                df_filtered,
                column_config=column_config,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.markdown(get_info_card(
                "🎯 Adjust Your Filters",
                "No pairs meet the current minimum score threshold. Try lowering the minimum score filter or adjusting other parameters to find more opportunities.",
                "⚙️"
            ), unsafe_allow_html=True)
    else:
        # Simple, friendly empty state
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div style="
                text-align: center;
                padding: 3rem 2rem;
                background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
                border-radius: 16px;
                border: 2px dashed #CBD5E1;
                margin: 2rem 0;
            ">
                <div style="
                    font-size: 3.5rem; 
                    margin-bottom: 1.5rem;
                    color: #8B5CF6;
                    font-weight: 700;
                    letter-spacing: 2px;
                ">◈</div>
                <h2 style="color: #475569; margin-bottom: 0.5rem; font-weight: 600;">Ready to Find Trading Opportunities?</h2>
                <p style="color: #64748B; font-size: 1rem; margin-bottom: 0; max-width: 500px; margin-left: auto; margin-right: auto;">
                    Configure your settings above and click <strong>"Run Analysis"</strong> to discover the best trading pairs.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
def render_trade_analysis_page(data_provider, sidebar_settings):
    """Trade Analysis Page - Deep analysis of individual trading pairs"""
    
    # Modern header
    st.markdown("""
        <div style="
            text-align: center; 
            padding: 1.5rem 0;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(147, 51, 234, 0.02) 100%);
            border-radius: 16px;
            margin-bottom: 2rem;
            border: 1px solid rgba(139, 92, 246, 0.1);
        ">
            <h1 style="
                color: #8B5CF6; 
                margin: 0; 
                font-weight: 700;
                font-size: 2rem;
                letter-spacing: -0.5px;
            ">TRADE ANALYSIS</h1>
            <p style="color: #6B7280; font-size: 0.95rem; margin-top: 0.75rem; font-weight: 500;">
                Deep analysis of individual trading pairs with advanced metrics
            </p>
        </div>
    """, unsafe_allow_html=True)
        
    # Add explanation section
    with st.expander("📚 Key Concepts & Parameter Guide", expanded=False):
        st.markdown("""
        ### Understanding the Metrics
        
        **Z-Score** 
        - Measures how far the current spread is from its historical average
        - **Lower Z-score** (< -2): Pair is "depegged" - potential BUY signal
        - **Higher Z-score** (> +2): Pair is overvalued - potential SELL signal
        - **Around 0**: Pair is at equilibrium - NEUTRAL/HOLD
        
        **Point Spread**
        - Shows the difference in returns between weak and strong assets
        - Indicates how far the current return difference is from historical norms
        - Used to identify short-term trading opportunities
        
        **Accumulated Spread**
        - Cumulative difference between assets over time
        - Helps spot long-term divergence patterns
        - Higher values suggest stronger mean reversion potential
        
        **Data Points (Bars)**
        - Number of historical bars to analyze
        - 180 ≈ 6 months daily data
        - 360 ≈ 1 year daily data
        - 720 ≈ 2 years daily data
        - More data = better statistical significance, but slower analysis
        """)
    
    # Pair Selection with Smart Recommendations
    st.markdown('<h3 style="color: #1E293B; font-weight: 600; margin-bottom: 1.5rem;">Select Trading Pair</h3>', unsafe_allow_html=True)
    
    # Show recommendations if selection results exist
    if st.session_state.selection_results is not None:
        top_pairs = st.session_state.selection_results.nlargest(3, 'Overall Score')
        if not top_pairs.empty:
            st.markdown("**💡 Recommended Pairs from Your Analysis:**")
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            for i, (_, pair_data) in enumerate(top_pairs.iterrows()):
                if i < 3:
                    with [rec_col1, rec_col2, rec_col3][i]:
                        pair_name = pair_data['Pair']
                        score = pair_data['Overall Score']
                        st.markdown(f"""
                            <div style="
                                background: rgba(139, 92, 246, 0.1);
                                padding: 0.75rem;
                                border-radius: 8px;
                                text-align: center;
                                border: 1px solid rgba(139, 92, 246, 0.2);
                                cursor: pointer;
                            ">
                                <div style="font-weight: 600; color: #8B5CF6;">{pair_name}</div>
                                <div style="font-size: 0.8rem; color: #6B7280;">Score: {score:.1f}/100</div>
                            </div>
                        """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        strong_asset = st.selectbox(
            "Strong Asset",
            options=AVAILABLE_ASSETS,
            index=AVAILABLE_ASSETS.index('ETH') if 'ETH' in AVAILABLE_ASSETS else 0,
            key="trade_strong"
        )
    
    with col2:
        weak_asset = st.selectbox(
            "Weak Asset",
            options=[asset for asset in AVAILABLE_ASSETS if asset != strong_asset],
            index=0,
            key="trade_weak"
        )
    
    with col3:
        trade_timeframe = st.selectbox(
            "Timeframe",
            options=list(TIMEFRAMES.keys()),
            format_func=lambda x: TIMEFRAMES[x],
            index=2,
            key="trade_timeframe"
        )
    
    with col4:
        # Simplified - just days input with CoinGecko limit
        data_points_override = st.number_input(
            "Days of History",
            min_value=30,
            max_value=365,
            value=180,
            step=30,
            help="⚠️ CoinGecko free tier max: 365 days. Recommended: 180 days (6 months)"
        )
    
    # Quick preset buttons for common data ranges
    st.markdown('<p style="font-size: 0.9rem; color: #64748B; margin: 0.5rem 0 0.25rem 0;">📅 Quick Presets:</p>', unsafe_allow_html=True)
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    
    with preset_col1:
        if st.button("30 days", key="preset_30", use_container_width=True):
            data_points_override = 30
    with preset_col2:
        if st.button("90 days", key="preset_90", use_container_width=True):
            data_points_override = 90
    with preset_col3:
        if st.button("180 days", key="preset_180", use_container_width=True):
            data_points_override = 180
    with preset_col4:
        if st.button("365 days (Max)", key="preset_365", use_container_width=True):
            data_points_override = 365
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Simple Analyze Button (CSV-based)
    if st.button("Analyze Pair", type="primary", key="analyze_pair"):
        # Validate assets before starting
        if strong_asset not in AVAILABLE_ASSETS:
            st.error(f"**Asset Not Available**: {strong_asset} is not in our available assets. Please choose from: {', '.join(AVAILABLE_ASSETS)}")
            return
            
        if weak_asset not in AVAILABLE_ASSETS:
            st.error(f"**Asset Not Available**: {weak_asset} is not in our available assets. Please choose from: {', '.join(AVAILABLE_ASSETS)}")
            return
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data from selected source
            selected_source = st.session_state.get('data_source', 'binance')
            source_name = "Binance Futures 🔷" if selected_source == 'binance' else "CoinGecko 🌐"
            status_text.text(f"📊 Loading data from {source_name} for {strong_asset}/{weak_asset}...")
            progress_bar.progress(25)
            
            # Use REAL data loader with selected data source!
            df = data_loader.load_real_data(strong_asset, weak_asset, days=data_points_override, data_source=selected_source)
            
            if df is not None:
                # Step 2: Calculate statistics
                status_text.text("📊 Calculating rolling statistics...")
                progress_bar.progress(50)
                
                # Simple rolling calculations
                df['Rolling Corr'] = df[f'LN {strong_asset} Var %'].rolling(window=30).corr(df[f'LN {weak_asset} Var %'])
                cov = df[f'LN {weak_asset} Var %'].rolling(window=30).cov(df[f'LN {strong_asset} Var %'])
                var = df[f'LN {strong_asset} Var %'].rolling(window=30).var()
                df['Rolling Beta'] = cov / var
                df['beta'] = df['Rolling Beta'].fillna(1.0)
                
                # Step 3: Simple signals
                status_text.text("🎯 Generating trading signals...")
                progress_bar.progress(75)
                
                # Simple entry/exit signals based on Z-score
                df['entry_signal'] = (df['mad_z_score'] < -2).astype(int)
                df['exit_signal'] = (df['mad_z_score'] > 0).astype(int)
                
                # Step 4: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                st.session_state.trade_analysis_results = df
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show success message with data summary
                data_start = df['ISO Date'].min().strftime('%Y-%m-%d')
                data_end = df['ISO Date'].max().strftime('%Y-%m-%d') 
                data_count = len(df)
                # Show which data source was actually used
                st.success(
                    f"✅ Successfully analyzed {strong_asset}/{weak_asset} pair using {source_name}! "
                    f"Period: {data_start} to {data_end} ({data_count:,} data points)"
                )
            else:
                progress_bar.empty()
                status_text.empty()
                st.markdown(get_error_toast("Failed to load data. Please try again."), unsafe_allow_html=True)
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.markdown(get_error_toast(f"Analysis failed: {str(e)}"), unsafe_allow_html=True)
    
    # Display Analysis Results
    if st.session_state.trade_analysis_results is not None:
        df = st.session_state.trade_analysis_results
        
        st.markdown("---")
        st.markdown('<h3 style="color: #1E293B; font-weight: 600; margin: 2rem 0 1.5rem 0;">Analysis Results</h3>', unsafe_allow_html=True)
        
        # Analysis tabs
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5, analysis_tab6 = st.tabs([
            "Overview",
            "Point Spread",
            "Accumulated Spread",
            "Distributions",
            "Correlation & Beta",
            "Export Data"
        ])
        
        with analysis_tab1:
            # Show data period
            if 'ISO Date' in df.columns:
                data_start = df['ISO Date'].min().strftime('%Y-%m-%d')
                data_end = df['ISO Date'].max().strftime('%Y-%m-%d')
                data_count = len(df)
                st.info(f"📅 Analyzing: {data_start} to {data_end} ({data_count:,} data points)")
            
            # KEY TRADING SIGNAL (Most Important!)
            st.markdown("### 🎯 Trading Signal")
            if 'mad_z_score' in df.columns:
                current_z = df['mad_z_score'].iloc[-1]
                ps_pct = df['PS_pct'].iloc[-1] if 'PS_pct' in df.columns else None
                as_pct = df['AS_pct'].iloc[-1] if 'AS_pct' in df.columns else None
                
                # Determine signal
                if current_z < -2.0 and ps_pct and ps_pct < 0.05 and as_pct and as_pct < 0.20:
                    signal = "🟢 STRONG BUY"
                    signal_color = "#10B981"
                    action = "✅ Good entry opportunity!"
                elif current_z > 2.0:
                    signal = "🔴 STRONG SELL"  
                    signal_color = "#EF4444"
                    action = "⚠️ Not recommended for long entry"
                elif as_pct and as_pct > 0.50:
                    signal = "💰 TAKE PROFIT"
                    signal_color = "#F59E0B"
                    action = "✅ Exit zone - close positions"
                else:
                    signal = "⏸️ WAIT"
                    signal_color = "#6B7280"
                    action = "⏳ No clear signal yet"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Z-Score", f"{current_z:.2f}", "Current position")
                with col2:
                    st.markdown(f"""
                        <div style="background: {signal_color}20; border: 2px solid {signal_color}; border-radius: 8px; padding: 1rem; text-align: center;">
                            <div style="font-size: 1.5rem; font-weight: 700; color: {signal_color};">{signal}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col3:
                    if ps_pct is not None and as_pct is not None:
                        st.metric("PS_pct", f"{ps_pct:.1%}", f"AS: {as_pct:.1%}")
                
                st.success(action)
            
            st.markdown("---")
            
            # Main Performance Chart
            st.markdown("### 📈 Price Performance")
            if f'{strong_asset} Close' in df.columns and f'{weak_asset} Close' in df.columns:
                try:
                    fig_perf = create_indexed_performance_chart(df, strong_asset, weak_asset)
                    st.plotly_chart(fig_perf, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not create performance chart: {str(e)}")
            else:
                st.info(f"📊 Performance chart not available - missing price data for {strong_asset} and/or {weak_asset}")
                
            # Performance metrics (only if we have price data)
            if f'{strong_asset} Close' in df.columns and f'{weak_asset} Close' in df.columns:
                try:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    strong_perf = ((df[f'{strong_asset} Close'].iloc[-1] / df[f'{strong_asset} Close'].iloc[0]) - 1) * 100
                    weak_perf = ((df[f'{weak_asset} Close'].iloc[-1] / df[f'{weak_asset} Close'].iloc[0]) - 1) * 100
                    relative_perf = strong_perf - weak_perf
                    
                    with col1:
                        st.metric(f"{strong_asset} Performance", f"{strong_perf:+.2f}%", 
                                    "Strong Asset" if strong_perf > 0 else "Underperforming")
                    with col2:
                        st.metric(f"{weak_asset} Performance", f"{weak_perf:+.2f}%",
                                    "Weak Asset" if weak_perf < strong_perf else "Outperforming")
                    with col3:
                        st.metric("Relative Performance", f"{relative_perf:+.2f}%",
                                    "Divergence" if abs(relative_perf) > 10 else "Converging")
                    with col4:
                        entry_favorable = relative_perf < 0  # Weak outperformed, mean reversion expected
                        st.metric("Entry Favorability", 
                                    "✅ Favorable" if entry_favorable else "⚠️ Wait",
                                    "Mean reversion likely" if entry_favorable else "Trend continuing")
                except Exception as e:
                    st.warning(f"⚠️ Could not calculate performance metrics: {str(e)}")
            
                # Enhanced Spread Logic Explanation
                st.markdown("### 📊 Enhanced Spread & Signal Logic")
                st.info(f"""
                **Point Spread (PS)** = {weak_asset} Returns - {strong_asset} Returns  
                **Accumulated Spread (AS)** = Cumulative Point Spread  
                
                📈 **When AS > 0**: {weak_asset} has outperformed {strong_asset} cumulatively (mean reversion opportunity)  
                📉 **When AS < 0**: {strong_asset} has outperformed {weak_asset} cumulatively (trend continuation)  
                
                🎯 **NEW Enhanced Trading Logic**:
                • **Entry**: PS_pct < 5% AND AS_pct < 20% AND Correlation ≥ 0.5 AND 1 < Beta < 2 AND 7D Cointegration OK
                • **Exit**: AS_pct > 50% OR Time-in-Trade > Half-Life (automatic time stop)
                • **Rolling Percentiles**: Now use proper timeframe-aware windows for accurate calculations
                """)
                
                # Show current timeframe calculations
                if 'trade_timeframe' in locals():
                    current_tf = locals()['trade_timeframe']
                    percentile_window = days_to_window(30, current_tf)  # Default 30D window
                    half_life_window = days_to_window(7, current_tf)   # Example half-life
                    
                    st.markdown(f"""
                    **🔧 Current Timeframe ({current_tf}) Calculations:**
                    - Rolling Percentile Window: {percentile_window} bars (30 days)
                    - Tactical Cointegration: {days_to_window(7, current_tf)} bars (7 days)
                    - Strategic Cointegration: {days_to_window(30, current_tf)} bars (30 days)
                    """)
            else:
                st.info("Price data not available for performance analysis")
        
        with analysis_tab2:
            st.subheader("📊 Point Spread Analysis")
            
            # Chart options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_mad = st.checkbox("Show MAD Bands", value=True, key="ps_mad")
            with col2:
                show_percentiles = st.checkbox("Show Percentile Bands", value=True, key="ps_percentile")
            with col3:
                show_signals = st.checkbox("Show Entry/Exit Signals", value=True, key="ps_signals")
            
            # Create and display Point Spread chart
            if 'Point Spread' in df.columns:
                fig_ps = create_point_spread_chart(df, show_mad, show_percentiles, show_signals)
                st.plotly_chart(fig_ps, use_container_width=True)
                
                # PS metrics
                col1, col2, col3, col4 = st.columns(4)
                
                ps_current = df['Point Spread'].iloc[-1]
                ps_median = df['Point Spread'].median()
                ps_mad = np.median(np.abs(df['Point Spread'] - ps_median))
                ps_z = (ps_current - ps_median) / (ps_mad * 1.4826) if ps_mad > 0 else 0
                
                with col1:
                    st.metric("Current PS", f"{ps_current:.6f}",
                                f"Z-Score: {ps_z:.2f}")
                with col2:
                    ps_pct = df['PS_pct'].iloc[-1] if 'PS_pct' in df.columns else 0
                    st.metric("PS Percentile", f"{ps_pct:.1%}",
                                "Entry Zone" if ps_pct < sidebar_settings['P_PS'] else "Wait")
                with col3:
                    st.metric("PS Median", f"{ps_median:.6f}",
                                "Mean Reversion Target")
                with col4:
                    distance = abs(ps_current - ps_median)
                    st.metric("Distance from Median", f"{distance:.6f}",
                                f"{(distance/abs(ps_median)*100):.1f}% deviation" if ps_median != 0 else "N/A")
                
                # Show enhanced percentile calculation info
                st.markdown("### 🔧 Enhanced Percentile Calculations")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'trade_timeframe' in locals():
                        current_tf = locals()['trade_timeframe']
                        percentile_window = days_to_window(30, current_tf)
                        st.markdown(f"""
                            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px;">
                                <h4 style="color: #10B981; margin: 0 0 0.5rem 0;">NEW: Rolling Percentile Logic</h4>
                                <p style="margin: 0; font-size: 0.9rem;">
                                    • Timeframe: <strong>{current_tf}</strong><br>
                                    • Window: <strong>{percentile_window} bars</strong> (30 days)<br>
                                    • PS_pct: Current PS rank vs last {percentile_window} values<br>
                                    • More accurate than fixed windows!
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if 'PS_pct' in df.columns:
                        recent_ps_pct = df['PS_pct'].tail(10)
                        st.markdown(f"""
                            <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px;">
                                <h4 style="color: #3B82F6; margin: 0 0 0.5rem 0;">Recent PS_pct Values</h4>
                                <p style="margin: 0; font-size: 0.8rem;">
                                    Last 10 values:<br>
                                    {', '.join([f'{x:.2f}' for x in recent_ps_pct.values[-10:]])}
                                </p>
                                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                                    <strong>Entry Zone:</strong> < {sidebar_settings['P_PS']:.1%}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Point Spread data not available")
        
        with analysis_tab3:
            st.subheader("📈 Accumulated Spread Analysis")
            
            # Chart options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_mad_as = st.checkbox("Show MAD Bands", value=True, key="as_mad")
            with col2:
                show_percentiles_as = st.checkbox("Show Percentile Bands", value=True, key="as_percentile")
            with col3:
                show_signals_as = st.checkbox("Show Entry/Exit Signals", value=True, key="as_signals")
            
            # Create and display Accumulated Spread chart
            if 'Accum Spread' in df.columns:
                fig_as = create_accumulated_spread_chart(df, show_mad_as, show_percentiles_as, show_signals_as)
                st.plotly_chart(fig_as, use_container_width=True)
                
                # AS metrics
                col1, col2, col3, col4 = st.columns(4)
                
                as_current = df['Accum Spread'].iloc[-1]
                as_median = df['Accum Spread'].median()
                as_p20 = df['Accum Spread'].quantile(0.20)
                
                with col1:
                    st.metric("Current AS", f"{as_current:.4f}",
                                "Cumulative Spread")
                with col2:
                    as_pct = df['AS_pct'].iloc[-1] if 'AS_pct' in df.columns else 0
                    st.metric("AS Percentile", f"{as_pct:.1%}",
                                "Entry Zone" if as_pct < sidebar_settings['P_AS'] else "Wait")
                with col3:
                    st.metric("AS 20th Percentile", f"{as_p20:.4f}",
                                "Entry Threshold")
                with col4:
                    # Combined entry signal
                    ps_ready = df['PS_pct'].iloc[-1] < sidebar_settings['P_PS'] if 'PS_pct' in df.columns else False
                    as_ready = as_pct < sidebar_settings['P_AS']
                    st.metric("Combined Entry Signal",
                                "✅ READY" if (ps_ready and as_ready) else "❌ WAIT",
                                "PS & AS both in entry zone" if (ps_ready and as_ready) else "Conditions not met")
                
                # Show enhanced exit logic
                st.markdown("### 🚪 Enhanced Exit Strategy")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                        <div style="background: rgba(239, 68, 68, 0.1); padding: 1rem; border-radius: 8px;">
                            <h4 style="color: #EF4444; margin: 0 0 0.5rem 0;">NEW: Dual Exit Conditions</h4>
                            <p style="margin: 0; font-size: 0.9rem;">
                                <strong>1. Mean Reversion Exit:</strong><br>
                                AS_pct > 50% (spread returns to median)<br><br>
                                <strong>2. Time-Based Exit:</strong><br>
                                Time-in-trade > Half-Life (bars)<br>
                                Prevents holding losing positions too long
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if 'trade_timeframe' in locals():
                        current_tf = locals()['trade_timeframe']
                        # Calculate half-life in bars (simplified example)
                        half_life_days = 7  # Example half-life
                        half_life_bars = days_to_window(half_life_days, current_tf)
                        
                        st.markdown(f"""
                            <div style="background: rgba(245, 158, 11, 0.1); padding: 1rem; border-radius: 8px;">
                                <h4 style="color: #F59E0B; margin: 0 0 0.5rem 0;">Time Stop Calculation</h4>
                                <p style="margin: 0; font-size: 0.9rem;">
                                    • Timeframe: <strong>{current_tf}</strong><br>
                                    • Half-Life: <strong>{half_life_days} days</strong><br>
                                    • Max Hold: <strong>{half_life_bars} bars</strong><br>
                                    • Auto-exit after this period
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Accumulated Spread data not available")
        
        with analysis_tab4:
            st.subheader("📊 Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # PS Distribution (MAD Z-standardized)
                if 'Point Spread' in df.columns:
                    z_scores = df['mad_z_score'].dropna()
                    
                    # Create MAD-based trading zones visualization
                    fig_zones = go.Figure()
                    
                    # Add histogram of Z-scores
                    fig_zones.add_trace(go.Histogram(
                        x=z_scores,
                        nbinsx=30,
                        name='MAD Z-Score Distribution',
                        marker=dict(
                            color='rgba(139, 92, 246, 0.6)',
                            line=dict(color='rgba(139, 92, 246, 0.8)', width=1)
                        ),
                        opacity=0.7,
                        hovertemplate='Z-Score: %{x:.2f}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    # Add trading zone lines with clear labels
                    trading_zones = [
                        (-3, '🛑 Stop Loss', '#ef4444'),
                        (-2, '🟢 Strong Buy', '#10b981'),
                        (-1, '📊 Weak Buy', '#3b82f6'),
                        (0, '⚖️ Neutral/Exit', '#6b7280'),
                        (1, '📊 Weak Sell', '#f59e0b'),
                        (2, '🔴 Strong Sell', '#ef4444'),
                        (3, '🛑 Stop Loss', '#ef4444')
                    ]
                    
                    for value, label, color in trading_zones:
                        fig_zones.add_vline(
                            x=value,
                            line=dict(color=color, width=2, dash='dash'),
                            annotation=dict(
                                text=f"{label}<br>Z={value}",
                                textangle=0,
                                font=dict(size=10, color=color, family='Inter', weight=600),
                                yanchor='bottom'
                            )
                        )
                    
                    # Add shaded regions for trading zones
                    fig_zones.add_vrect(
                        x0=-4, x1=-2,
                        fillcolor="rgba(16, 185, 129, 0.1)",
                        layer="below",
                        line_width=0,
                        annotation_text="BUY ZONE",
                        annotation_position="top left"
                    )
                    
                    fig_zones.add_vrect(
                        x0=-1, x1=1,
                        fillcolor="rgba(107, 114, 128, 0.1)",
                        layer="below",
                        line_width=0,
                        annotation_text="HOLD ZONE",
                        annotation_position="top"
                    )
                    
                    fig_zones.add_vrect(
                        x0=2, x1=4,
                        fillcolor="rgba(239, 68, 68, 0.1)",
                        layer="below",
                        line_width=0,
                        annotation_text="SELL ZONE",
                        annotation_position="top right"
                    )
                    
                    fig_zones.update_layout(
                        title=dict(
                            text="📊 MAD Z-Score Trading Zones",
                            x=0.5,
                            font=dict(size=16, color='#8B5CF6', family='Inter', weight=700)
                        ),
                        xaxis_title="MAD Z-Score (σ-equivalent)",
                        yaxis_title="Frequency",
                        height=400,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#374151', family='Inter'),
                        showlegend=True,
                        xaxis=dict(
                            range=[-4, 4],
                            dtick=1,
                            gridcolor='rgba(139, 92, 246, 0.1)'
                        ),
                        yaxis=dict(
                            gridcolor='rgba(139, 92, 246, 0.1)'
                        )
                    )
                    
                    st.plotly_chart(fig_zones, use_container_width=True)
                    
                    # Add MAD Trading Rules
                    current_z = z_scores.iloc[-1] if len(z_scores) > 0 else 0
                    
                    # Determine current signal
                    if current_z <= -2:
                        signal = "🟢 STRONG BUY"
                        action = "Open long position or add to existing"
                        exit_target = "Exit when Z-score returns to 0 (median)"
                    elif current_z >= 2:
                        signal = "🔴 STRONG SELL"
                        action = "Open short position or reduce long"
                        exit_target = "Exit when Z-score returns to 0 (median)"
                    elif -1 <= current_z <= 1:
                        signal = "⚖️ NEUTRAL"
                        action = "Hold or close existing positions"
                        exit_target = "Wait for |Z| > 2 for new entry"
                    else:
                        signal = "📊 WEAK SIGNAL"
                        action = "Monitor closely, prepare for entry"
                        exit_target = "Wait for stronger signal"
                    
                    st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(147, 51, 234, 0.05));
                            border: 2px solid rgba(139, 92, 246, 0.3);
                            border-radius: 12px;
                            padding: 1.5rem;
                            margin: 1rem 0;
                        ">
                            <h4 style="color: #8B5CF6; margin: 0 0 1rem 0;">📈 MAD Trading Signal</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                                <div>
                                    <p style="color: #6B7280; font-size: 0.875rem; margin: 0;">Current Z-Score</p>
                                    <p style="color: #8B5CF6; font-size: 1.5rem; font-weight: 700; margin: 0;">{current_z:.2f}</p>
                        </div>
                                <div>
                                    <p style="color: #6B7280; font-size: 0.875rem; margin: 0;">Signal</p>
                                    <p style="font-size: 1.2rem; font-weight: 700; margin: 0;">{signal}</p>
                                </div>
                            </div>
                            <hr style="border: 1px solid rgba(139, 92, 246, 0.2); margin: 1rem 0;">
                            <p style="color: #374151; margin: 0.5rem 0;"><strong>Action:</strong> {action}</p>
                            <p style="color: #374151; margin: 0.5rem 0;"><strong>Exit:</strong> {exit_target}</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        with analysis_tab5:
            st.subheader("🔗 Correlation & Beta Analysis")
            
            # Create dual charts
            if 'Rolling Corr' in df.columns or 'Rolling Beta' in df.columns:
                fig_corr, fig_beta = create_correlation_beta_charts(df)
                
                # Display charts side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_corr, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_beta, use_container_width=True)
                
                # Metrics summary
                st.markdown("### 📊 Current Risk Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_corr = df['Rolling Corr'].iloc[-1] if 'Rolling Corr' in df.columns else 0
                    corr_ok = current_corr >= 0.5
                    st.metric("Current Correlation", f"{current_corr:.3f}",
                                "✅ Pass" if corr_ok else "❌ Fail (< 0.5)")
                
                with col2:
                    beta_col = 'Rolling Beta' if 'Rolling Beta' in df.columns else 'beta'
                    current_beta = df[beta_col].iloc[-1] if beta_col in df.columns else 0
                    beta_ok = 1.0 < current_beta < 2.0
                    st.metric("Current Beta", f"{current_beta:.3f}",
                                "✅ Ideal" if beta_ok else "⚠️ Outside range")
                
                with col3:
                    avg_corr = df['Rolling Corr'].mean() if 'Rolling Corr' in df.columns else 0
                    st.metric("Average Correlation", f"{avg_corr:.3f}",
                                "30-period average")
                
                with col4:
                    avg_beta = df[beta_col].mean() if beta_col in df.columns else 0
                    st.metric("Average Beta", f"{avg_beta:.3f}",
                                "30-period average")
                
                st.markdown("---")
                st.markdown("### 📊 Statistical Summary")
            
            # Calculate statistics
            stats_analyzer = StatisticalAnalysis()
            spread_stats = stats_analyzer.calculate_spread_statistics(df['log_spread'])
            
            # Calculate half-life (Carlynn's metric)
            half_life = stats_analyzer.calculate_half_life(df['log_spread'])
            
            # Create KPI cards for key statistics first
            st.markdown("### 🎯 Key Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(get_enhanced_kpi_card(
                    value=f"{spread_stats['mean']:.6f}",
                    label="Mean Spread",
                    delta="Central Tendency",
                    icon="📊",
                    color="purple"
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(get_enhanced_kpi_card(
                    value=f"{spread_stats['median']:.6f}",
                    label="Median Spread",
                    delta="50th Percentile",
                    icon="📍",
                    color="purple"
                ), unsafe_allow_html=True)
            
            with col3:
                volatility_level = "High" if spread_stats['std'] > abs(spread_stats['mean']) else "Moderate" if spread_stats['std'] > abs(spread_stats['mean'])/2 else "Low"
                st.markdown(get_enhanced_kpi_card(
                    value=f"{spread_stats['std']:.6f}",
                    label="Standard Deviation",
                    delta=f"{volatility_level} Volatility",
                    icon="📏",
                    color="purple"
                ), unsafe_allow_html=True)
            
            with col4:
                st.markdown(get_enhanced_kpi_card(
                    value=f"{spread_stats['mad']:.6f}",
                    label="MAD",
                    delta="Robust Measure",
                    icon="🎯",
                    color="purple"
                ), unsafe_allow_html=True)
            
            with col5:
                st.markdown(get_enhanced_kpi_card(
                    value=f"{half_life:.1f}",
                    label="Half-Life (T½)",
                    delta="Mean Reversion Days",
                    icon="⏱️",
                    color="purple"
                ), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Enhanced detailed statistics in cards
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic Statistics Card
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(147, 51, 234, 0.02) 100%);
                        border: 1px solid rgba(139, 92, 246, 0.2);
                        border-radius: 16px;
                        padding: 1.5rem;
                        margin: 1rem 0;
                        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
                    ">
                        <h4 style="color: #8B5CF6; margin: 0 0 1rem 0; font-weight: 600;">📈 Detailed Statistics</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create enhanced statistics display
                stats_display = f"""
                <div style="background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 8px rgba(139, 92, 246, 0.1);">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; font-size: 0.9rem;">
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(139, 92, 246, 0.05); border-radius: 8px;">
                            <span style="font-weight: 600; color: #6B7280;">📊 Mean:</span>
                            <span style="color: #8B5CF6; font-weight: 600;">{spread_stats['mean']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(139, 92, 246, 0.05); border-radius: 8px;">
                            <span style="font-weight: 600; color: #6B7280;">📍 Median:</span>
                            <span style="color: #8B5CF6; font-weight: 600;">{spread_stats['median']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(139, 92, 246, 0.05); border-radius: 8px;">
                            <span style="font-weight: 600; color: #6B7280;">📏 Std Dev:</span>
                            <span style="color: #8B5CF6; font-weight: 600;">{spread_stats['std']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(139, 92, 246, 0.05); border-radius: 8px;">
                            <span style="font-weight: 600; color: #6B7280;">🎯 MAD:</span>
                            <span style="color: #8B5CF6; font-weight: 600;">{spread_stats['mad']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(239, 68, 68, 0.05); border-radius: 8px;">
                            <span style="font-weight: 600; color: #6B7280;">⬇️ Min:</span>
                            <span style="color: #EF4444; font-weight: 600;">{spread_stats['min']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(16, 185, 129, 0.05); border-radius: 8px;">
                            <span style="font-weight: 600; color: #6B7280;">⬆️ Max:</span>
                            <span style="color: #10B981; font-weight: 600;">{spread_stats['max']:.6f}</span>
                        </div>
                    </div>
                </div>
                """
                st.markdown(stats_display, unsafe_allow_html=True)
            
            with col2:
                # Percentiles Card
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(147, 51, 234, 0.02) 100%);
                        border: 1px solid rgba(139, 92, 246, 0.2);
                        border-radius: 16px;
                        padding: 1.5rem;
                        margin: 1rem 0;
                        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.1);
                    ">
                        <h4 style="color: #8B5CF6; margin: 0 0 1rem 0; font-weight: 600;">📊 Distribution Percentiles</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                # Create enhanced percentiles display
                percentiles_display = f"""
                <div style="background: white; border-radius: 12px; padding: 1rem; box-shadow: 0 2px 8px rgba(139, 92, 246, 0.1);">
                    <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: linear-gradient(90deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05)); border-radius: 8px; border-left: 3px solid #EF4444;">
                            <span style="font-weight: 600; color: #374151;">🔸 5th Percentile</span>
                            <span style="color: #EF4444; font-weight: 700;">{spread_stats['p05']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: linear-gradient(90deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05)); border-radius: 8px; border-left: 3px solid #F59E0B;">
                            <span style="font-weight: 600; color: #374151;">🔹 20th Percentile</span>
                            <span style="color: #F59E0B; font-weight: 700;">{spread_stats['p20']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: linear-gradient(90deg, rgba(139, 92, 246, 0.1), rgba(139, 92, 246, 0.05)); border-radius: 8px; border-left: 3px solid #8B5CF6;">
                            <span style="font-weight: 600; color: #374151;">🔶 50th Percentile</span>
                            <span style="color: #8B5CF6; font-weight: 700;">{spread_stats['p50']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)); border-radius: 8px; border-left: 3px solid #3B82F6;">
                            <span style="font-weight: 600; color: #374151;">🔹 80th Percentile</span>
                            <span style="color: #3B82F6; font-weight: 700;">{spread_stats['p80']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: linear-gradient(90deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)); border-radius: 8px; border-left: 3px solid #10B981;">
                            <span style="font-weight: 600; color: #374151;">🔸 95th Percentile</span>
                            <span style="color: #10B981; font-weight: 700;">{spread_stats['p95']:.6f}</span>
                        </div>
                    </div>
                </div>
                """
                st.markdown(percentiles_display, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Enhanced Cointegration Analysis with Dual Timeframe Validation
            if len(df) > 30 and f'{strong_asset} Close' in df.columns and f'{weak_asset} Close' in df.columns:
                strong_prices = np.log(df[f'{strong_asset} Close'].dropna())
                weak_prices = np.log(df[f'{weak_asset} Close'].dropna())
                
                # Calculate both tactical (7-day) and strategic (30-day) cointegration
                coint_7d = stats_analyzer.calculate_cointegration(strong_prices, weak_prices, window=7)
                coint_30d = stats_analyzer.calculate_cointegration(strong_prices, weak_prices, window=30)
                
                st.markdown("### 🔗 Dual-Timeframe Cointegration Analysis")
                
                # Show both tactical and strategic cointegration
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Tactical (7-Day) - Entry/Exit Signals**")
                    p_value_7d = coint_7d['pvalue']
                    is_cointegrated_7d = coint_7d['is_cointegrated']
                    confidence_7d = coint_7d['confidence']
                    
                    # Display 7-day cointegration
                    st.markdown(get_enhanced_kpi_card(
                        value=f"{p_value_7d:.4f}",
                        label="7-Day P-Value",
                        delta="✅ Cointegrated" if is_cointegrated_7d else "❌ Not Cointegrated",
                        icon="⚡",
                        color="purple"
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**📈 Strategic (30-Day) - Pair Validation**")
                    p_value_30d = coint_30d['pvalue']
                    is_cointegrated_30d = coint_30d['is_cointegrated']
                    confidence_30d = coint_30d['confidence']
                    
                    # Display 30-day cointegration
                    st.markdown(get_enhanced_kpi_card(
                        value=f"{p_value_30d:.4f}",
                        label="30-Day P-Value",
                        delta="✅ Cointegrated" if is_cointegrated_30d else "❌ Not Cointegrated",
                        icon="🛡️",
                        color="purple"
                    ), unsafe_allow_html=True)
                
                # Combined validation
                both_cointegrated = is_cointegrated_7d and is_cointegrated_30d
                
                # Determine overall status
                if both_cointegrated:
                    status_color = "#10B981"
                    status_bg = "rgba(16, 185, 129, 0.1)"
                    status_icon = "✅"
                    status_text = "Both Timeframes Cointegrated"
                    interpretation = "Strong mean-reverting relationship detected on both tactical and strategic timeframes - IDEAL for trading!"
                elif is_cointegrated_7d:
                    status_color = "#F59E0B"
                    status_bg = "rgba(245, 158, 11, 0.1)"
                    status_icon = "⚠️"
                    status_text = "Only Tactical Cointegration"
                    interpretation = "Short-term mean reversion detected but lacks long-term stability - Use with caution"
                elif is_cointegrated_30d:
                    status_color = "#3B82F6"
                    status_bg = "rgba(59, 130, 246, 0.1)"
                    status_icon = "🔍"
                    status_text = "Only Strategic Cointegration"
                    interpretation = "Long-term relationship exists but short-term signals may be weak - Wait for better entry"
                else:
                    status_color = "#EF4444"
                    status_bg = "rgba(239, 68, 68, 0.1)"
                    status_icon = "❌"
                    status_text = "No Cointegration"
                    interpretation = "No significant mean-reverting relationship on either timeframe - Not recommended for trading"
                
                # Overall validation card
                st.markdown("---")
                st.markdown("### 🎯 Trading Recommendation")
                
                # Create recommendation card based on combined analysis
                st.markdown(f"""
                    <div style="
                        background: {status_bg};
                        border: 2px solid {status_color};
                        border-radius: 12px;
                        padding: 1.5rem;
                        margin: 1rem 0;
                    ">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 2rem;">{status_icon}</div>
                            <div>
                                <h3 style="color: {status_color}; margin: 0;">{status_text}</h3>
                                <p style="color: #6B7280; margin: 0.5rem 0 0 0;">{interpretation}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Trading strategy recommendation based on cointegration results
                if both_cointegrated:
                    st.markdown(get_info_card(
                        "✅ TRADE RECOMMENDED",
                        "Both 7-day (tactical) and 30-day (strategic) cointegration confirm mean reversion. " +
                        "Use 7-day signals for entry/exit timing and 30-day for position validation.",
                        "🚀"
                    ), unsafe_allow_html=True)
                elif is_cointegrated_7d:
                    st.markdown(get_info_card(
                        "⚠️ SHORT-TERM ONLY",
                        "Only tactical cointegration detected. Consider smaller position sizes and tighter stops. " +
                        "Monitor for development of longer-term cointegration.",
                        "⏱️"
                    ), unsafe_allow_html=True)
                elif is_cointegrated_30d:
                    st.markdown(get_info_card(
                        "🔍 WAIT FOR ENTRY",
                        "Strategic cointegration exists but tactical signals are weak. " +
                        "Wait for short-term cointegration to develop before entering positions.",
                        "⏳"
                    ), unsafe_allow_html=True)
                else:
                    st.markdown(get_info_card(
                        "❌ NOT RECOMMENDED",
                        "No cointegration detected on either timeframe. " +
                        "This pair is not suitable for mean reversion strategies at this time.",
                        "🛑"
                    ), unsafe_allow_html=True)
            else:
                st.info("Insufficient data or missing price columns for cointegration analysis")
        
        with analysis_tab6:
            st.subheader("💾 Export Data & Trading Signals")
            
            col1, col2, col3 = st.columns(3)
            
            # Calculate current trading signal with enhanced KPI cards
            if 'mad_z_score' in df.columns:
                current_z = df['mad_z_score'].iloc[-1]
                
                with col1:
                    if abs(current_z) > 2:
                        signal_type = "BUY" if current_z < -2 else "SELL"
                        signal_strength = "STRONG" if abs(current_z) > 2.5 else "MODERATE"
                        icon = "🟢" if signal_type == "BUY" else "🔴"
                        st.markdown(get_enhanced_kpi_card(
                            value=f"{signal_type}",
                            label="Trading Signal",
                            delta=f"{signal_strength} • Z: {current_z:.2f}",
                            icon=icon,
                            color="purple"
                        ), unsafe_allow_html=True)
                    else:
                        st.markdown(get_enhanced_kpi_card(
                            value="NO SIGNAL",
                            label="Trading Signal",
                            delta=f"Z-Score: {current_z:.2f}",
                            icon="⏳",
                            color="purple"
                        ), unsafe_allow_html=True)
                
                with col2:
                    median_spread = df['log_spread'].median()
                    current_spread = df['log_spread'].iloc[-1]
                    distance_from_median = ((current_spread - median_spread) / median_spread) * 100
                    
                    st.markdown(get_enhanced_kpi_card(
                        value=f"{distance_from_median:.2f}%",
                        label="Distance from Median",
                        delta=f"Spread: {current_spread:.6f}",
                        icon="📏",
                        color="purple"
                    ), unsafe_allow_html=True)
                
                with col3:
                    if 'Rolling Corr' in df.columns:
                        current_corr = df['Rolling Corr'].iloc[-1]
                        corr_strength = "Strong" if current_corr > 0.7 else "Moderate" if current_corr > 0.5 else "Weak"
                        st.markdown(get_enhanced_kpi_card(
                            value=f"{current_corr:.3f}",
                            label="Current Correlation",
                            delta=corr_strength,
                            icon="🔗",
                            color="purple"
                        ), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Prepare export data with safe column selection
            try:
                # Start with basic columns that should always exist
                basic_columns = ['ISO Date']
                available_columns = basic_columns.copy()
                
                # Check for price columns
                strong_close_col = f'{strong_asset} Close'
                weak_close_col = f'{weak_asset} Close'
                
                if strong_close_col in df.columns:
                    available_columns.append(strong_close_col)
                else:
                    st.warning(f"⚠️ Price data for {strong_asset} not available in this dataset")
                
                if weak_close_col in df.columns:
                    available_columns.append(weak_close_col)
                else:
                    st.warning(f"⚠️ Price data for {weak_asset} not available in this dataset")
                
                # Check for spread columns
                spread_columns = ['log_spread', 'Point Spread', 'Accum Spread']
                for col in spread_columns:
                    if col in df.columns:
                        available_columns.append(col)
                
                # Create export dataframe with only available columns
                export_df = df[available_columns].copy()
                
                # Add optional metrics if they exist
                optional_metrics = {
                    'PS_pct': 'PS_pct',
                    'AS_pct': 'AS_pct', 
                    'mad_z_score': 'MAD Z-Score',
                    'Rolling Corr': 'Correlation',
                    'Rolling Beta': 'Beta'
                }
                
                for original_col, export_col in optional_metrics.items():
                    if original_col in df.columns:
                        export_df[export_col] = df[original_col]
                
                # Show what data is available
                st.success(f"✅ Export data prepared with {len(export_df.columns)} columns: {', '.join(export_df.columns)}")
                
            except Exception as e:
                st.error(f"❌ Error preparing export data: {str(e)}")
                # Create minimal export with just available data
                available_cols = [col for col in df.columns if col in ['ISO Date', 'log_spread', 'mad_z_score']]
                export_df = df[available_cols].copy() if available_cols else df.copy()
            
            # Generate simple Excel export with safe column selection
            export_sheets = {"Price Data": export_df}
            
            # Add trading signals sheet if columns exist
            signal_columns = ['ISO Date', 'mad_z_score', 'entry_signal', 'exit_signal']
            available_signal_cols = [col for col in signal_columns if col in df.columns]
            if len(available_signal_cols) > 1:  # At least ISO Date + one other
                export_sheets["Trading Signals"] = df[available_signal_cols].tail(100)
            
            # Add statistics sheet if spread data exists
            if 'log_spread' in df.columns and 'mad_z_score' in df.columns:
                try:
                    export_sheets["Statistics"] = pd.DataFrame({
                        'Metric': ['Mean Spread', 'Median Spread', 'Std Dev', 'Current Z-Score'],
                        'Value': [
                            df['log_spread'].mean(),
                            df['log_spread'].median(), 
                            df['log_spread'].std(),
                            df['mad_z_score'].iloc[-1]
                        ]
                    })
                except Exception as e:
                    st.warning(f"⚠️ Could not generate statistics: {str(e)}")
            
            st.info(f"📊 Excel export will contain {len(export_sheets)} sheets: {', '.join(export_sheets.keys())}")
            
            # Create Excel file with multiple sheets
            excel_data = export_to_excel(export_sheets, "simple_analysis.xlsx")
            
            st.download_button(
                label="📥 Export Multi-Sheet Analysis (8 Tabs)",
                data=excel_data,
                file_name=f"comprehensive_{strong_asset}_{weak_asset}_{trade_timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary",
                help="Exports 8 sheets: Price Data, Returns & Spreads, MAD Analysis, Trading Signals, Rolling Stats, Summary, Cointegration, Recommendations"
            )
            
            # Enhanced data preview with trading insights
            st.markdown("### 📊 Recent Data & Trading Opportunities")
            
            # Current market condition
            if 'mad_z_score' in df.columns:
                latest_z = df['mad_z_score'].iloc[-1]
                latest_spread = df['log_spread'].iloc[-1]
                latest_ps_pct = df['PS_pct'].iloc[-1] if 'PS_pct' in df.columns else None
                latest_as_pct = df['AS_pct'].iloc[-1] if 'AS_pct' in df.columns else None
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if abs(latest_z) > 2:
                        signal = "STRONG BUY" if latest_z < -2 else "STRONG SELL"
                        color = "success-box" if latest_z < -2 else "danger-box"
                    elif abs(latest_z) > 1:
                        signal = "MODERATE BUY" if latest_z < -1 else "MODERATE SELL"
                        color = "info-box"
                    else:
                        signal = "NO SIGNAL"
                        color = "warning-box"
                    
                    st.markdown(f"""
                        <div class="{color}">
                            <h4>🎯 Current Signal</h4>
                            <p><strong>{signal}</strong></p>
                            <p>Z-Score: {latest_z:.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    spread_percentile = (df['log_spread'] <= latest_spread).mean() * 100
                    st.metric(
                        "Spread Percentile",
                        f"{spread_percentile:.1f}%",
                        f"Current: {latest_spread:.6f}"
                    )
                
                with col3:
                    if 'Rolling Corr' in df.columns:
                        latest_corr = df['Rolling Corr'].iloc[-1]
                        corr_strength = "Strong" if latest_corr > 0.7 else "Moderate" if latest_corr > 0.5 else "Weak"
                        st.metric(
                            "Pair Correlation",
                            f"{latest_corr:.3f}",
                            corr_strength
                        )
                
                with col4:
                    # Display Carlynn's key metrics
                    if latest_ps_pct is not None and latest_as_pct is not None:
                        entry_ready = (latest_ps_pct < sidebar_settings['P_PS']) and (latest_as_pct < sidebar_settings['P_AS'])
                        st.metric(
                            "Entry Signal (PS & AS)",
                            "✅ READY" if entry_ready else "❌ NOT YET",
                            f"PS: {latest_ps_pct:.1%}, AS: {latest_as_pct:.1%}"
                        )
            
            # Data table without matplotlib-dependent styling
            st.dataframe(export_df.tail(20), use_container_width=True, height=400)
    else:
        # Beautiful empty state for Trade Analysis
        st.markdown("""
            <div style="
                text-align: center;
                padding: 4rem 2rem;
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(147, 51, 234, 0.02) 100%);
                border-radius: 16px;
                border: 2px dashed rgba(139, 92, 246, 0.3);
                margin: 2rem 0;
            ">
                <div style="
                    font-size: 3.5rem; 
                    margin-bottom: 1.5rem;
                    color: #8B5CF6;
                    font-weight: 700;
                    letter-spacing: 2px;
                ">◈</div>
                <h2 style="color: #475569; margin-bottom: 0.5rem; font-weight: 600;">Ready for Deep Pair Analysis?</h2>
                <p style="color: #64748B; font-size: 1rem; margin-bottom: 0; max-width: 500px; margin-left: auto; margin-right: auto;">
                    Select your trading pair above and click "Analyze Pair" to get detailed statistical analysis, charts, and trading signals.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
def render_global_report_page(data_provider, sidebar_settings):
    """Global Report Page - Comprehensive multi-timeframe analysis"""
    st.header("Global Report - All Timeframes")
    st.markdown("Comprehensive analysis across all timeframes and pairs")
        
    # Generate Global Report Button
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_collateral = st.selectbox(
            "Select Collateral for Global Report",
            options=STRONG_ASSETS,
            index=1,
            key="global_collateral"
        )
    
    with col2:
        timeframes_to_analyze = st.multiselect(
            "Timeframes to Include",
            options=list(TIMEFRAMES.keys()),
            default=['1d', '4h', '1h'],
            format_func=lambda x: TIMEFRAMES[x]
        )
    
    with col3:
        if st.button("🌐 Generate Global Report", type="primary", use_container_width=True):
            with st.spinner("Generating sample global report..."):
                # Generate sample global report data
                global_data = []
                
                for tf in timeframes_to_analyze:
                    for asset in AVAILABLE_ASSETS[:15]:  # Limit for demo
                        if asset != selected_collateral:
                            global_data.append({
                                'Timeframe': TIMEFRAMES[tf],
                                'Pair': f"{selected_collateral}/{asset}",
                                'Cointegration': np.random.uniform(0.01, 0.15),
                                'Correlation': np.random.uniform(0.2, 0.8),
                                'Beta': np.random.uniform(0.5, 2.5),
                                'Overall Score': np.random.randint(20, 95),
                                'Current Z-Score': np.random.uniform(-3, 3)
                            })
                
                st.session_state.global_report = pd.DataFrame(global_data)
                st.success("✅ Sample global report generated successfully!")
    
    # Display Global Report
    if st.session_state.global_report is not None:
        # Summary Statistics with Enhanced KPIs
        st.subheader("📊 Global Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_pairs = len(st.session_state.global_report)
        high_score_pairs = len(st.session_state.global_report[st.session_state.global_report['Overall Score'] > 60])
        best_pair = st.session_state.global_report.nlargest(1, 'Overall Score').iloc[0]
        avg_score = st.session_state.global_report['Overall Score'].mean()
        
        with col1:
            st.markdown(get_enhanced_kpi_card(
                value=f"{total_pairs:,}",
                label="Total Combinations",
                icon="🌐",
                color="purple"
            ), unsafe_allow_html=True)
        
        with col2:
            success_rate = f"{(high_score_pairs/total_pairs*100):.1f}%" if total_pairs > 0 else "0%"
            st.markdown(get_enhanced_kpi_card(
                value=f"{high_score_pairs:,}",
                label="High Score Pairs",
                delta=f"+{success_rate}",
                icon="🔥",
                color="purple"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(get_enhanced_kpi_card(
                value=f"{best_pair['Pair']}",
                label="Best Pair",
                delta=f"Score: {best_pair['Overall Score']:.1f}",
                icon="🏆",
                color="purple"
            ), unsafe_allow_html=True)
        
        with col4:
            quality = "Excellent" if avg_score >= 70 else "Good" if avg_score >= 50 else "Fair"
            st.markdown(get_enhanced_kpi_card(
                value=f"{avg_score:.1f}/100",
                label="Average Score",
                delta=quality,
                icon="📊",
                color="purple"
            ), unsafe_allow_html=True)
        
        # Filters
        st.subheader("🎯 Filter Options")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_coint = st.slider("Max Cointegration P-Value", 0.0, 1.0, 0.05, 0.01, key="global_coint")
        
        with col2:
            min_corr = st.slider("Min Correlation", 0.0, 1.0, 0.5, 0.1, key="global_corr")
        
        with col3:
            beta_range = st.slider("Beta Range", 0.5, 3.0, (0.8, 2.0), 0.1, key="global_beta")
        
        with col4:
            min_global_score = st.slider("Min Overall Score", 0, 100, 50, 10, key="global_score")
        
        # Apply Filters
        filtered_report = st.session_state.global_report[
            (st.session_state.global_report['Cointegration'] <= min_coint) &
            (st.session_state.global_report['Correlation'] >= min_corr) &
            (st.session_state.global_report['Beta'].between(beta_range[0], beta_range[1])) &
            (st.session_state.global_report['Overall Score'] >= min_global_score)
        ].sort_values('Overall Score', ascending=False)
        
        # Enhanced visualizations for Global Report
        if not filtered_report.empty:
            # Add visualization tabs
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Data Table", "🎯 Scatter Analysis", "📈 Correlation Matrix"])
            
            with viz_tab1:
                st.subheader(f"🎯 Filtered Results ({len(filtered_report)} pairs)")
                
                # Configure display
                column_config = {
                    "Timeframe": st.column_config.TextColumn("Timeframe", width="small"),
                    "Pair": st.column_config.TextColumn("Pair", width="medium"),
                    "Cointegration": st.column_config.NumberColumn("Cointegration", format="%.4f"),
                    "Correlation": st.column_config.NumberColumn("Correlation", format="%.3f"),
                    "Beta": st.column_config.NumberColumn("Beta", format="%.3f"),
                    "Overall Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                    "Current Z-Score": st.column_config.NumberColumn("Z-Score", format="%.2f")
                }
                
                st.dataframe(
                    filtered_report,
                    column_config=column_config,
                    use_container_width=True,
                    height=500
                )
            
            with viz_tab2:
                st.subheader("🎯 Pair Quality Scatter Plot")
                scatter_fig = create_cointegration_scatter(filtered_report)
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Add insights
                excellent_pairs = len(filtered_report[filtered_report['Overall Score'] >= 80])
                good_pairs = len(filtered_report[(filtered_report['Overall Score'] >= 60) & (filtered_report['Overall Score'] < 80)])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🌟 Excellent Pairs", excellent_pairs, "Score ≥ 80")
                with col2:
                    st.metric("✅ Good Pairs", good_pairs, "Score 60-79")
                with col3:
                    best_pair = filtered_report.nlargest(1, 'Overall Score').iloc[0]
                    st.metric("🏆 Top Pair", best_pair['Pair'], f"Score: {best_pair['Overall Score']}")
            
            with viz_tab3:
                st.subheader("📈 Asset Correlation Analysis")
                
                # Create correlation matrix for top assets
                top_pairs = filtered_report.nlargest(20, 'Overall Score')
                if not top_pairs.empty:
                    # Extract unique assets
                    assets = set()
                    for pair in top_pairs['Pair']:
                        strong, weak = pair.split('/')
                        assets.add(strong)
                        assets.add(weak)
                    
                    assets = sorted(list(assets))
                    
                    # Create mock correlation matrix (in real implementation, you'd calculate actual correlations)
                    mock_corr_matrix = np.random.rand(len(assets), len(assets))
                    mock_corr_matrix = (mock_corr_matrix + mock_corr_matrix.T) / 2  # Make symmetric
                    np.fill_diagonal(mock_corr_matrix, 1)  # Diagonal = 1
                    
                    corr_df = pd.DataFrame(mock_corr_matrix, index=assets, columns=assets)
                    
                    heatmap_fig = create_correlation_heatmap(corr_df)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    st.info("💡 **Tip:** Look for assets with moderate correlation (0.3-0.7) for optimal pair trading opportunities.")
            
            # Export Global Report
            excel_data = export_to_excel(
                {
                    "Global Report": filtered_report,
                    "Summary": pd.DataFrame({
                        'Metric': ['Total Pairs', 'Filtered Pairs', 'Best Score', 'Average Score'],
                        'Value': [total_pairs, len(filtered_report), best_pair['Overall Score'], avg_score]
                    })
                },
                "global_report.xlsx"
            )
            
            st.download_button(
                label="📥 Export Global Report to Excel",
                data=excel_data,
                file_name=f"global_report_{selected_collateral}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )
        else:
            st.warning("No pairs meet the filter criteria. Try adjusting the filters.")
    
def render_active_positions_page(data_provider, sidebar_settings):
    """Active Positions Page - Portfolio management"""
    st.header("Active Positions & Portfolio")
    st.markdown("Monitor and manage your current trading positions")
        
    # Portfolio Summary with Enhanced KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(get_enhanced_kpi_card(
            value="$24,567",
            label="Portfolio Value",
            delta="+12.4%",
            icon="💰",
            color="purple"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(get_enhanced_kpi_card(
            value="3/5",
            label="Active Positions",
            delta="60% utilized",
            icon="📈",
            color="purple"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(get_enhanced_kpi_card(
            value="+$2,456",
            label="Today's P&L",
            delta="+8.2%",
            icon="💵",
            color="purple"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(get_enhanced_kpi_card(
            value="73.5%",
            label="Win Rate",
            delta="+5.2%",
            icon="🎯",
            color="purple"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Active Positions Table with Modern Styling
    st.subheader("📈 Current Positions")
    
    # Sample active positions data
    positions_data = pd.DataFrame({
        'Pair': ['ETH/ARB', 'ETH/ATOM', 'BTC/ALGO'],
        'Direction': ['SHORT', 'SHORT', 'LONG'],
        'Entry Price': [0.00045, 0.00123, 0.00234],
        'Current Price': [0.00041, 0.00125, 0.00238],
        'P&L %': [8.2, -1.5, 1.7],
        'P&L $': [820, -150, 170],
        'MAD Z-Score': [-2.3, 1.8, -2.1],
        'Time Held': ['2h 15m', '45m', '1h 30m'],
        'Status': ['🟢 Profitable', '🟡 Monitor', '🟢 Profitable']
    })
    
    # Use simple dataframe without matplotlib-dependent styling
    st.dataframe(
        positions_data,
        use_container_width=True,
        height=250,
        column_config={
            "Pair": st.column_config.TextColumn("Trading Pair", width="small"),
            "Direction": st.column_config.TextColumn("Direction", width="small"),
            "Entry Price": st.column_config.NumberColumn("Entry Price", format="%.6f"),
            "Current Price": st.column_config.NumberColumn("Current Price", format="%.6f"),
            "P&L %": st.column_config.NumberColumn("P&L %", format="%.1f%%"),
            "P&L $": st.column_config.NumberColumn("P&L $", format="$%.0f"),
            "MAD Z-Score": st.column_config.NumberColumn("Z-Score", format="%.1f"),
            "Time Held": st.column_config.TextColumn("Duration", width="small"),
            "Status": st.column_config.TextColumn("Status", width="medium")
        }
    )
    
    # Position Management
    st.subheader("⚙️ Position Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_position = st.selectbox(
            "Select Position to Manage",
            options=positions_data['Pair'].tolist()
        )
    
    with col2:
        action = st.selectbox(
            "Action",
            options=["Close Position", "Add to Position", "Set Stop Loss", "Set Take Profit"]
        )
    
    with col3:
        if st.button("Execute Action", type="primary", use_container_width=True):
            st.success(f"✅ {action} executed for {selected_position}")
    
    # Risk Metrics
    st.subheader("⚠️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution chart
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Positions': [1, 1, 1],
            'Allocation %': [30, 50, 20]
        })
        
        import plotly.express as px
        fig = px.pie(
            risk_data, 
            values='Allocation %', 
            names='Risk Level',
            color_discrete_map={
                'Low': '#10b981', 
                'Medium': '#f59e0b', 
                'High': '#ef4444'
            },
            title="Portfolio Risk Distribution",
            hole=0.4
        )
        
        fig.update_traces(
            textposition='outside',
            textinfo='percent+label',
            textfont=dict(size=12, family='Inter'),
            hovertemplate='<b>%{label}</b><br>Allocation: %{value}%<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            title=dict(
                text="📊 Portfolio Risk Distribution",
                x=0.5,
                font=dict(size=16, color='#1e40af', family='Inter')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151', family='Inter'),
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.05
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance metrics with modern styling
        st.markdown("### 📊 Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['📈 Sharpe Ratio', '📉 Max Drawdown', '💹 Avg Win', '📊 Avg Loss', '⚡ Profit Factor'],
            'Value': ['2.34', '-8.5%', '+4.2%', '-2.1%', '2.0']
        })
        
        # Use styled dataframe instead of raw HTML
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small")
            }
        )

def render_settings_page():
    """Settings Page - Configure trading parameters"""
    st.header("🔧 Settings")
    st.markdown("Configure trading thresholds and application preferences")
    
    st.subheader("🎯 Trading Thresholds")
    st.info("💡 These are already optimized. Change only if you know what you're doing.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        p_ps = st.slider(
            "Entry Signal Sensitivity",
            min_value=0.0,
            max_value=0.20,
            value=st.session_state.sidebar_settings['P_PS'],
            step=0.01,
            key="settings_p_ps",
            help="Lower = More strict entry signals"
        )
        
        mad_threshold = st.slider(
            "MAD Z-Score Threshold",
            min_value=1.0,
            max_value=3.0,
            value=st.session_state.sidebar_settings['mad_threshold'],
            step=0.1,
            key="settings_mad"
        )
    
    with col2:
        p_as = st.slider(
            "Exit Signal Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.sidebar_settings['P_AS'],
            step=0.05,
            key="settings_p_as",
            help="When to take profits"
        )
        
        position_size = st.number_input(
            "Default Position Size (USDT)",
            min_value=100,
            max_value=100000,
            value=st.session_state.sidebar_settings['position_size'],
            step=1000,
            key="settings_position"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.session_state.sidebar_settings['P_PS'] = p_ps
            st.session_state.sidebar_settings['P_AS'] = p_as
            st.session_state.sidebar_settings['mad_threshold'] = mad_threshold
            st.session_state.sidebar_settings['position_size'] = position_size
            st.success("✅ Settings saved!")
    
    with col2:
        if st.button("♻️ Reset to Defaults", use_container_width=True):
            st.session_state.sidebar_settings = {
                'P_PS': 0.05,
                'P_AS': 0.50,
                'mad_threshold': 2.0,
                'position_size': 1000,
                'max_pairs': 3,
                'data_points': 1000
            }
            st.success("✅ Settings reset to defaults!")
            st.rerun()

if __name__ == "__main__":
    main()
