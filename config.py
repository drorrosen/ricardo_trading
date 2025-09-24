"""
Configuration file for Pair Trading Dashboard
Contains all constants and settings
"""

# Binance Loans Available Assets (as of 2024)
BINANCE_LOANS_ASSETS = [
    'BTC', 'ETH', 'BNB', 'ADA', 'ALGO', 'APE', 'APT', 'ARB', 'ARKM', 'ATOM',
    'AVAX', 'AXS', 'BCH', 'BLUR', 'BNX', 'CFX', 'CHZ', 'COMP', 'CRV', 'DOGE',
    'DOT', 'DYDX', 'EGLD', 'ENJ', 'EOS', 'ETC', 'FET', 'FIL', 'FLOKI', 'FLOW',
    'FTM', 'GALA', 'GMT', 'GMX', 'GRT', 'HBAR', 'ICP', 'ID', 'IMX', 'INJ',
    'JTO', 'KAVA', 'KDA', 'KLAY', 'LDO', 'LINK', 'LRC', 'LTC', 'LUNA', 'LUNC',
    'MAGIC', 'MANA', 'MATIC', 'MEME', 'MINA', 'MKR', 'NEAR', 'NEO', 'OP', 'ORDI',
    'PENDLE', 'PEPE', 'QNT', 'RDNT', 'RNDR', 'ROSE', 'RUNE', 'SAND', 'SEI', 'SHIB',
    'SNX', 'SOL', 'SSV', 'STG', 'STX', 'SUI', 'SUSHI', 'TIA', 'TRB', 'TRX',
    'TWT', 'UNI', 'VET', 'WAVES', 'WLD', 'WOO', 'XLM', 'XRP', 'XTZ', 'YGG', 'ZIL'
]

# Strong assets commonly used as collateral
STRONG_ASSETS = ['BTC', 'ETH', 'BNB', 'SOL']

# Timeframe options - Extended for comprehensive analysis
TIMEFRAMES = {
    '1m': '1 Minute',     # Ultra high frequency
    '3m': '3 Minutes',    # Short-term scalping
    '5m': '5 Minutes',    # Day trading
    '15m': '15 Minutes',  # Intraday analysis
    '30m': '30 Minutes',  # Short-term swing
    '1h': '1 Hour',       # Hourly analysis (recommended)
    '2h': '2 Hours',      # Extended intraday
    '4h': '4 Hours',      # Multi-session analysis
    '6h': '6 Hours',      # Quarter-day analysis
    '8h': '8 Hours',      # Third-day analysis
    '12h': '12 Hours',    # Half-day analysis
    '1d': '1 Day',        # Daily analysis (standard)
    '3d': '3 Days',       # Multi-day trends
    '1w': '1 Week',       # Weekly analysis
    '1M': '1 Month'       # Monthly analysis (long-term)
}

# Fibonacci-based period recommendations (days: {timeframe: candles})
FIBONACCI_PERIODS = {
    5: {'5m': 1440, '15m': 480, '1h': 120},     # 5 days
    8: {'15m': 768, '1h': 192, '4h': 48},       # 8 days
    13: {'15m': 1248, '1h': 312, '4h': 78},     # 13 days
    21: {'1h': 504, '4h': 126, '1d': 21},       # 21 days (recommended)
    34: {'4h': 204, '1d': 34},                  # 34 days
    55: {'4h': 330, '1d': 55},                  # 55 days
    89: {'1d': 89}                               # 89 days
}

# Default settings for analysis
DEFAULT_SETTINGS = {
    'timeframe': '1d',
    'cointegration_window': 30,
    'correlation_window': 30,
    'beta_window': 30,
    'cointegration_pvalue_threshold': 0.05,
    'correlation_threshold': 0.5,
    'beta_min': 1.0,
    'beta_max': 2.0,
    'mad_window': 120,
    'mad_k_entry': 2.0,
    'mad_k_exit': 0.5,
    'use_ewma_beta': False,
    'ewma_decay_factor': 0.94,
    'strong_coin': 'ETH',
    'weak_coin': 'BCH',
    'base_capital': 10000,
    'max_ltv': 0.75,
    'data_points': 1000,  # Increased default for longer analysis
    'P_PS': 0.05,  
    'P_AS': 0.20,
    'max_data_points': 5000,  # Maximum allowed data points
    'min_data_points': 100    # Minimum required data points
}

# Score thresholds
SCORE_THRESHOLDS = {
    'strong': 70,
    'moderate': 50,
    'weak': 30
}

# Statistical thresholds
STATISTICAL_THRESHOLDS = {
    'cointegration': {
        'strong': 0.01,
        'moderate': 0.05,
        'weak': 0.10
    },
    'correlation': {
        'strong': 0.7,
        'moderate': 0.5,
        'weak': 0.3
    },
    'beta': {
        'ideal_min': 1.0,
        'ideal_max': 2.0,
        'acceptable_min': 0.8,
        'acceptable_max': 2.5
    }
}

# MAD (Median Absolute Deviation) settings
MAD_SETTINGS = {
    'conversion_factor': 1.4826,  # Convert MAD to standard deviation equivalent
    'entry_threshold': -2.0,      # Enter when Z-score < -2
    'exit_threshold': 0.0,         # Exit when Z-score crosses median
    'stop_loss_threshold': 3.0    # Stop loss at 3 MAD
}

# Chart settings
CHART_SETTINGS = {
    'height': 500,
    'colors': {
        'primary': '#6366f1',
        'secondary': '#8b5cf6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#3b82f6'
    },
    'histogram_bins': 50,
    'line_width': 2,
    'marker_size': 10
}

# Export settings
EXPORT_SETTINGS = {
    'excel_engine': 'xlsxwriter',
    'date_format': '%Y%m%d_%H%M',
    'float_format': '%.6f'
}

# API settings
API_SETTINGS = {
    'binance_base_url': 'https://api.binance.com/api/v3',
    'rate_limit_delay': 0.1,  # seconds between API calls
    'max_retries': 3,
    'timeout': 30
}

# Cache settings (in seconds)
CACHE_SETTINGS = {
    'price_data': 300,      # 5 minutes
    'analysis_results': 600, # 10 minutes
    'static_data': 3600     # 1 hour
}
