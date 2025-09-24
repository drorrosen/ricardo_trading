"""
Backend Logic for Pair Trading Dashboard
=======================================

This module contains all the computational logic, data processing, and business rules
for the pair trading application. The frontend (app_modular.py) should only handle
UI components and call these backend functions.

Structure:
- BinanceDataFetcher: Handles all API calls to Binance
- PairDataProcessor: Processes and transforms trading pair data
- StatisticalAnalysis: All statistical calculations and tests
- TradingSignalGenerator: Generates buy/sell signals based on strategies
- PairAnalyzer: Main orchestrator for pair analysis
- DashboardDataProvider: Provides all data for dashboard components
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import coint
import requests
import time
from typing import Dict, List, Tuple, Optional, Any

# ---------- Helpers: timeframe-to-candles & rolling percentile ----------

BARS_PER_DAY = {
    '1m': 60*24,
    '5m': 12*24,
    '15m': 4*24,
    '30m': 2*24,
    '1h': 24,
    '4h': 6,
    '1d': 1,
    '1w': 1/7,   # careful: weekly means ~1/7 “days” per bar; not used for 7D/30D windows
}

def days_to_window(days: int, timeframe: str) -> int:
    """Convert day horizon to #bars for a given timeframe (e.g., 7D on 1h => 168)."""
    per_day = BARS_PER_DAY.get(timeframe, 24)  # default 1h-like
    window = int(max(10, round(days * per_day)))
    return window

def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile of the *current* value vs the last window values.
    Example: at t, percentile = rank of series[t] among series[t-window+1:t] / window.
    """
    out = pd.Series(index=series.index, dtype=float)
    if window <= 1:
        return out

    vals = series.values
    for i in range(len(series)):
        if i < window - 1 or pd.isna(vals[i]):
            out.iat[i] = np.nan
            continue
        w = vals[i - window + 1: i + 1]
        x = vals[i]
        # rank of x among the window
        rank = np.sum(w < x) + 0.5 * np.sum(w == x)
        out.iat[i] = rank / window
    return out


class BinanceDataFetcher:
    """Handles all Binance API interactions"""
    
    BASE_URL = 'https://api.binance.com/api/v3'
    
    @staticmethod
    def fetch_klines(symbol: str, interval: str = '1d', limit: int = 500, 
                    start_time: Optional[int] = None, end_time: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Fetch historical kline data from Binance with proper pagination."""
        try:
            url = f'{BinanceDataFetcher.BASE_URL}/klines'
            params = {
                'symbol': f'{symbol}USDT',
                'interval': interval,
            }
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            all_rows = []
            remaining = limit
            last_end = end_time

            while remaining > 0:
                # clone params per loop
                q = dict(params)
                q['limit'] = min(remaining, 1000)

                # paginate forward by startTime if no endTime, else paginate backward by endTime
                if last_end is not None:
                    q['endTime'] = last_end

                resp = requests.get(url, params=q, timeout=15)
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break

                all_rows.extend(batch)

                # update pagination cursor
                if last_end is not None:
                    # walk backward
                    oldest_open_time = batch[0][0]
                    last_end = oldest_open_time - 1
                else:
                    # walk forward
                    newest_open_time = batch[-1][0]
                    params['startTime'] = newest_open_time + 1

                remaining -= len(batch)
                time.sleep(0.08)

                # stop if Binance returns less than requested (no more data)
                if len(batch) < q['limit']:
                    break

            if not all_rows:
                return None

            df = pd.DataFrame(all_rows, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                'buy_quote_volume', 'ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            num_cols = ['open', 'high', 'low', 'close', 'volume']
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=['open_time', 'close']).drop_duplicates('open_time').sort_values('open_time')
            df = df.rename(columns={'open_time': 'timestamp'})
            df = df.reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None


class PairDataProcessor:
    """Processes pair trading data"""
    
    @staticmethod
    def fetch_pair_data(strong_asset: str, weak_asset: str, 
                       interval: str = '1d', limit: int = 500,
                       start_date: Optional[pd.Timestamp] = None,
                       end_date: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """Fetch and process data for a trading pair with optional date filtering"""
        fetcher = BinanceDataFetcher()
        
        # Convert dates to timestamps if provided
        start_time = None
        end_time = None
        if start_date:
            start_time = int(start_date.timestamp() * 1000)
        if end_date:
            end_time = int(end_date.timestamp() * 1000)
        
        strong_df = fetcher.fetch_klines(strong_asset, interval, limit, start_time, end_time)
        weak_df = fetcher.fetch_klines(weak_asset, interval, limit, start_time, end_time)
        
        if strong_df is None or weak_df is None:
            return None
        
        # Merge dataframes
        # --- replace the simple assignment with a safe inner merge by timestamp ---
        strong_df = strong_df[['timestamp', 'close']].rename(columns={'close': f'{strong_asset} Close'})
        weak_df   = weak_df[['timestamp', 'close']].rename(columns={'close': f'{weak_asset} Close'})

        df = pd.merge(
            strong_df, weak_df, on='timestamp', how='inner', suffixes=('', '')
        ).sort_values('timestamp').reset_index(drop=True)

        df = df.rename(columns={'timestamp': 'ISO Date'})

                
        # Calculate log prices and returns (following Carlynn's logic)
        df[f'LN {strong_asset} Close'] = np.log(df[f'{strong_asset} Close'])
        df[f'LN {weak_asset} Close'] = np.log(df[f'{weak_asset} Close'])
        
        df[f'LN {strong_asset} Var %'] = df[f'LN {strong_asset} Close'].diff()
        df[f'LN {weak_asset} Var %'] = df[f'LN {weak_asset} Close'].diff()
        
        # Calculate spreads (Carlynn's PS and AS)
        # Point Spread = difference in returns (weak - strong)
        # When PS > 0: weak asset outperforming strong asset (mean reversion opportunity)
        # When PS < 0: strong asset outperforming weak asset (trend continuation)
        df['Point Spread'] = df[f'LN {weak_asset} Var %'] - df[f'LN {strong_asset} Var %']
        df['Accum Spread'] = df['Point Spread'].cumsum()
        df['log_spread'] = df[f'LN {weak_asset} Close'] - df[f'LN {strong_asset} Close']
        
        # Save timeframe used so downstream functions can derive window sizes (7D/30D)
        try:
            df.attrs['timeframe'] = interval
        except Exception:
            pass

        
        return df


class StatisticalAnalysis:
    """Contains all statistical analysis functions"""
    
    @staticmethod
    def calculate_cointegration(price1: pd.Series, price2: pd.Series, 
                               window: Optional[int] = None) -> Dict[str, Any]:
        """Calculate Engle-Granger cointegration test"""
        if window is not None and len(price1) > window:
            price1 = price1.tail(window)
            price2 = price2.tail(window)
        
        try:
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
    
    @staticmethod
    def calculate_rolling_cointegration(price1: pd.Series, price2: pd.Series, 
                                       window: int = 252) -> pd.Series:
        """Calculate rolling cointegration p-values"""
        pvalues = []
        dates = []
        
        for i in range(window, len(price1)):
            window_price1 = price1.iloc[i-window:i]
            window_price2 = price2.iloc[i-window:i]
            
            result = StatisticalAnalysis.calculate_cointegration(window_price1, window_price2)
            pvalues.append(result['pvalue'])
            dates.append(price1.index[i])
        
        return pd.Series(pvalues, index=dates)
    
    @staticmethod
    def calculate_correlation(returns1: pd.Series, returns2: pd.Series, 
                            window: int = 30) -> float:
        """Calculate correlation between returns"""
        if len(returns1) >= window:
            return returns1.tail(window).corr(returns2.tail(window))
        return 0.0
    
    @staticmethod
    def calculate_beta(returns1: pd.Series, returns2: pd.Series, 
                      window: int = 30) -> float:
        """Calculate beta (volatility ratio)"""
        if len(returns1) >= window:
            cov = returns2.tail(window).cov(returns1.tail(window))
            var = returns1.tail(window).var()
            return cov / var if var != 0 else 0.0
        return 0.0
    
    @staticmethod
    def calculate_ewma_beta(returns1: pd.Series, returns2: pd.Series, 
                           decay_factor: float = 0.94) -> pd.Series:
        """Calculate EWMA (Exponentially Weighted Moving Average) Beta"""
        ewma_var1 = returns1.ewm(alpha=1-decay_factor).var()
        ewma_cov = returns1.ewm(alpha=1-decay_factor).cov(returns2)
        ewma_beta = ewma_cov / ewma_var1
        return ewma_beta
    
    @staticmethod
    def calculate_mad_z_scores(series: pd.Series, window: int = 120) -> Tuple[pd.Series, pd.Series, pd.Series]:
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
        
        return (pd.Series(mad_z_scores, index=series.index), 
                pd.Series(medians, index=series.index), 
                pd.Series(mads, index=series.index))
    
    @staticmethod
    def calculate_spread_statistics(spread_series: pd.Series, window: int = 120) -> Dict[str, float]:
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
    
    @staticmethod
    def coint_windows_for_timeframe(timeframe: str) -> Tuple[int, int]:
       """Return (win_7d, win_30d) in bars for a given timeframe."""
       return days_to_window(7, timeframe), days_to_window(30, timeframe)

    
    @staticmethod
    def calculate_half_life(spread_series: pd.Series) -> float:
        """Calculate half-life of mean reversion using OLS"""
        spread_lag = spread_series.shift(1)
        spread_diff = spread_series - spread_lag
        spread_lag = spread_lag[1:]
        spread_diff = spread_diff[1:]
        
        # OLS regression
        X = spread_lag.values.reshape(-1, 1)
        y = spread_diff.values
        
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            half_life = -np.log(2) / beta[1] if beta[1] < 0 else np.inf
            return max(1, min(half_life, 365))  # Cap between 1 and 365 days
        except:
            return 60  # Default half-life


class TradingSignalGenerator:
    """Generates trading signals based on various criteria"""
    
    @staticmethod
    def generate_signals(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate trading signals using rolling percentiles (PS_pct, AS_pct) + risk gates.
        Entry: PS_pct < P_PS AND AS_pct < P_AS AND corr>=0.5 AND 1<beta<2 AND cointegration_7d OK
        Exit:  AS_pct > 0.50  OR  time-in-trade > half-life (bars)
        """
        # --- choose windows ---
        timeframe = settings.get('timeframe', '1h')
        # default rolling window: tie to 30 days (or use half-life later if you pass it)
        roll_days = settings.get('percentile_days', 30)
        roll_win  = days_to_window(roll_days, timeframe)

        # --- ensure spreads present ---
        if 'Point Spread' not in df.columns or 'Accum Spread' not in df.columns:
            raise ValueError("DataFrame must contain 'Point Spread' and 'Accum Spread' columns.")

        # --- rolling percentiles for PS and AS ---
        df['PS_pct'] = rolling_percentile(df['Point Spread'], roll_win)
        df['AS_pct'] = rolling_percentile(df['Accum Spread'], roll_win)

        # --- MAD z-score on log_spread (for context) ---
        mad_window = settings.get('mad_window', max(60, roll_win))
        z, med, mad = StatisticalAnalysis.calculate_mad_z_scores(df['log_spread'], window=mad_window)
        df['mad_z_score'] = z
        df['spread_median'] = med
        df['spread_mad'] = mad

        # --- rolling correlation/beta if not present (fallback) ---
        if 'Rolling Corr' not in df.columns or 'Rolling Beta' not in df.columns:
            rwin = settings.get('corr_beta_window', 30)
            strong = settings.get('strong_coin', None)
            weak   = settings.get('weak_coin', None)
            if strong and weak:
                rs = df[f'LN {strong} Var %']
                rw = df[f'LN {weak} Var %']
                df['Rolling Corr'] = rs.rolling(rwin).corr(rw)
                cov = rw.rolling(rwin).cov(rs)
                var = rs.rolling(rwin).var()
                df['Rolling Beta'] = cov / var
            else:
                df['Rolling Corr'] = np.nan
                df['Rolling Beta'] = np.nan
        df['beta'] = df['Rolling Beta']

        # --- cointegration 7D check (pass in from caller if already computed) ---
        coint_7d_ok = settings.get('cointegration_7d_ok', None)
        if coint_7d_ok is None:
            # optional quick check using last win7 bars
            strong = settings.get('strong_coin', None)
            weak   = settings.get('weak_coin', None)
            if strong and weak:
                win7 = days_to_window(7, timeframe)
                p1 = np.log(df[f'{strong} Close']).tail(win7).dropna()
                p2 = np.log(df[f'{weak} Close']).tail(win7).dropna()
                c = StatisticalAnalysis.calculate_cointegration(p1, p2)
                coint_7d_ok = c.get('is_cointegrated', False)
            else:
                coint_7d_ok = False

        # --- thresholds ---
        p_ps = settings.get('P_PS', 0.05)
        p_as = settings.get('P_AS', 0.20)

        corr_ok = df['Rolling Corr'] >= 0.5
        beta_ok = (df['beta'] > 1.0) & (df['beta'] < 2.0)
        gates_ok = corr_ok & beta_ok & coint_7d_ok

        # --- entry condition ---
        entry = (df['PS_pct'] < p_ps) & (df['AS_pct'] < p_as) & gates_ok
        df['entry_signal'] = entry.astype(int)

        # --- time stop using half-life ---
        # compute half-life on log_spread
        hl_days = StatisticalAnalysis.calculate_half_life(df['log_spread'])  # returns *days*
        hl_bars = max(1, days_to_window(int(np.ceil(hl_days)), timeframe))

        # compute exit as: AS_pct > 0.5 or time stop
        # to track “time in trade”, create a simple counter that increments after an entry until exit:
        time_in_trade = np.zeros(len(df), dtype=int)
        in_pos = False
        counter = 0
        for i in range(len(df)):
            if entry.iat[i] and not in_pos:
                in_pos = True
                counter = 0
            if in_pos:
                counter += 1
            time_in_trade[i] = counter
            # if we later set exit, in_pos will be cleared in second pass

        as_exit = df['AS_pct'] > 0.50
        time_exit = pd.Series(time_in_trade, index=df.index) > hl_bars

        # provisional exit
        provisional_exit = (as_exit | time_exit)

        # finalize position lifecycle (exit resets the timer)
        exit_signal = np.zeros(len(df), dtype=int)
        in_pos = False
        for i in range(len(df)):
            if entry.iat[i] and not in_pos:
                in_pos = True
            elif in_pos and provisional_exit.iat[i]:
                exit_signal[i] = 1
                in_pos = False

        df['exit_signal'] = exit_signal
        return df



class PairAnalyzer:
    """Main analyzer for pair trading opportunities"""
    
    def __init__(self, binance_assets: List[str]):
        self.binance_assets = binance_assets
        self.processor = PairDataProcessor()
        self.stats = StatisticalAnalysis()
        self.signal_gen = TradingSignalGenerator()
    
    def analyze_all_pairs(self, strong_asset: str, timeframe: str, 
                         window_coint: int, window_corr: int, 
                         window_beta: int) -> pd.DataFrame:
        """Analyze all weak assets against a strong asset"""
        results = []
        weak_assets = [asset for asset in self.binance_assets if asset != strong_asset]
        
        for weak_asset in weak_assets:
            df = self.processor.fetch_pair_data(strong_asset, weak_asset, timeframe)
            
            if df is None or len(df) < max(window_coint, window_corr, window_beta):
                continue
            
            try:
                # Calculate metrics
                result = self._analyze_single_pair(
                    df, strong_asset, weak_asset, timeframe,
                    window_coint, window_corr, window_beta
                )
                results.append(result)
            except Exception as e:
                continue
            
            # Rate limit protection
            time.sleep(0.1)
        
        return pd.DataFrame(results)
    
    def _analyze_single_pair(self, df: pd.DataFrame, strong_asset: str, 
                            weak_asset: str, timeframe: str,
                            window_coint: int, window_corr: int, 
                            window_beta: int) -> Dict[str, Any]:
        """Analyze a single pair using Carlynn's logic"""
        # Cointegration
        strong_prices = np.log(df[f'{strong_asset} Close'].dropna())
        weak_prices = np.log(df[f'{weak_asset} Close'].dropna())
        
        if len(strong_prices) >= window_coint:
            coint_result = self.stats.calculate_cointegration(
                strong_prices.tail(window_coint), 
                weak_prices.tail(window_coint)
            )
            pvalue = coint_result['pvalue']
        else:
            pvalue = 1.0
        
        # Correlation
        strong_returns = df[f'LN {strong_asset} Var %'].dropna()
        weak_returns = df[f'LN {weak_asset} Var %'].dropna()
        correlation = self.stats.calculate_correlation(strong_returns, weak_returns, window_corr)
        
        # Beta
        beta = self.stats.calculate_beta(strong_returns, weak_returns, window_beta)
        
        # Get PS_pct and AS_pct (Carlynn's key metrics)
        ps_pct = df['PS_pct'].iloc[-1] if 'PS_pct' in df.columns else None
        as_pct = df['AS_pct'].iloc[-1] if 'AS_pct' in df.columns else None
        
        # Calculate Carlynn's Opportunity Score
        score = self._calculate_opportunity_score(pvalue, correlation, beta, ps_pct, as_pct)
        
        # Current spread metrics
        current_spread = df['log_spread'].iloc[-1]
        spread_mean = df['log_spread'].mean()
        spread_std = df['log_spread'].std()
        z_score = (current_spread - spread_mean) / spread_std if spread_std != 0 else 0
        
        # Carlynn's entry condition check
        entry_ok = False
        if ps_pct is not None and as_pct is not None:
            entry_ok = (ps_pct < 0.05) and (as_pct < 0.20)  # Default thresholds
        
        return {
            'Timeframe': timeframe,
            'Strong Asset': strong_asset,
            'Weak Asset': weak_asset,
            'Pair': f'{strong_asset}/{weak_asset}',
            'Cointegration': round(pvalue, 4),
            'Correlation': round(correlation, 3),
            'Beta': round(beta, 3),
            'PS_pct': round(ps_pct, 3) if ps_pct else None,
            'AS_pct': round(as_pct, 3) if as_pct else None,
            'Overall Score': score,
            'Current Z-Score': round(z_score, 2),
            'Entry Signal': entry_ok,
            'Status': '✅' if score >= 60 else '⚠️' if score >= 40 else '❌'
        }
    
    def _calculate_opportunity_score(self, pvalue: float, correlation: float, beta: float, 
                                    ps_pct: Optional[float], as_pct: Optional[float]) -> float:
        """Calculate opportunity score using Carlynn's formula"""
        opp = 0.0
        
        # PS and AS components (main drivers of opportunity)
        if ps_pct is not None:
            opp += (0.5 - ps_pct)
        if as_pct is not None:
            opp += (0.5 - as_pct)
        
        # Boolean flags (Carlynn's logic)
        f_coin = (pvalue < 0.05)  # Cointegration OK
        f_corr = (correlation >= 0.5)  # Correlation OK
        f_beta = (1.0 < beta < 2.0)  # Beta OK
        
        # Add bonus points for passing thresholds
        opp += (0.2 if f_corr else 0)
        opp += (0.2 if f_beta else 0)
        opp += (0.2 if f_coin else 0)
        
        # Convert to percentage score (0-100)
        return min(max(opp * 100, 0), 100)


class DashboardDataProvider:
    """
    Provides all data and computations for dashboard components
    
    This class acts as the main interface between the frontend and backend,
    providing clean, formatted data for all dashboard components including
    KPIs, charts, tables, and analysis results.
    """
    
    def __init__(self, binance_assets: List[str]):
        """
        Initialize the dashboard data provider
        
        Args:
            binance_assets: List of available Binance assets for trading
        """
        self.binance_assets = binance_assets
        self.analyzer = PairAnalyzer(binance_assets)
        self.processor = PairDataProcessor()
        self.stats = StatisticalAnalysis()
        self.signal_gen = TradingSignalGenerator()
    
    def get_dashboard_overview_data(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get all data needed for the dashboard overview section
        
        Args:
            session_state: Streamlit session state containing analysis results
            
        Returns:
            Dictionary containing all overview metrics and status indicators
        """
        # Calculate basic metrics
        total_assets = len(self.binance_assets)
        possible_pairs = (total_assets * (total_assets - 1)) // 2
        
        # Get analysis status
        analysis_count = len(session_state.get('selection_results', [])) if session_state.get('selection_results') is not None else 0
        signal_count = len(session_state.get('active_signals', []))
        
        # Calculate system status
        has_selection = session_state.get('selection_results') is not None
        has_trade = session_state.get('trade_analysis_results') is not None
        has_global = session_state.get('global_report') is not None
        
        return {
            'kpis': {
                'market_status': {
                    'value': '24/7',
                    'label': 'Market Status',
                    'delta': '🟢 Active',
                    'icon': '🌐'
                },
                'available_assets': {
                    'value': str(total_assets),
                    'label': 'Available Assets',
                    'delta': 'Binance Loans',
                    'icon': '💎'
                },
                'possible_pairs': {
                    'value': f'{possible_pairs:,}',
                    'label': 'Possible Pairs',
                    'delta': 'Combinations',
                    'icon': '🔗'
                },
                'analyzed_pairs': {
                    'value': str(analysis_count),
                    'label': 'Analyzed Pairs',
                    'delta': 'Ready to Trade' if analysis_count > 0 else 'Run Analysis',
                    'icon': '📊'
                },
                'active_signals': {
                    'value': str(signal_count),
                    'label': 'Active Signals',
                    'delta': '+2 Today' if signal_count > 0 else 'No Signals',
                    'icon': '🎯'
                }
            },
            'system_status': {
                'selection_filter': {
                    'status': '✅ Complete' if has_selection else '⏳ Pending',
                    'color': 'success-box' if has_selection else 'warning-box',
                    'label': 'Selection Filter'
                },
                'trade_analysis': {
                    'status': '✅ Complete' if has_trade else '⏳ Pending',
                    'color': 'success-box' if has_trade else 'warning-box',
                    'label': 'Trade Analysis'
                },
                'global_report': {
                    'status': '✅ Complete' if has_global else '⏳ Pending',
                    'color': 'success-box' if has_global else 'warning-box',
                    'label': 'Global Report'
                },
                'trading_signals': {
                    'status': f'🎯 {signal_count} Active' if signal_count > 0 else '😴 No Signals',
                    'color': 'info-box' if signal_count > 0 else 'warning-box',
                    'label': 'Trading Signals'
                }
            }
        }
    
    def run_pair_selection_analysis(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Run the complete pair selection analysis
        
        Args:
            config: Configuration dictionary containing:
                - timeframe: Trading timeframe (e.g., '1h', '1d')
                - collateral: Strong asset to use as collateral
                - cointegration_window: Window for cointegration test
                - correlation_window: Window for correlation calculation
                - beta_window: Window for beta calculation
                - min_score: Minimum score filter
                
        Returns:
            DataFrame with analysis results for all pairs
        """
        try:
            # Extract configuration
            timeframe = config.get('timeframe', '1d')
            collateral = config.get('collateral', 'ETH')
            coint_window = config.get('cointegration_window', 30)
            corr_window = config.get('correlation_window', 30)
            beta_window = config.get('beta_window', 30)
            min_score = config.get('min_score', 40)
            
            # Run analysis
            results_df = self.analyzer.analyze_all_pairs(
                collateral, timeframe, coint_window, corr_window, beta_window
            )
            
            # Apply score filter
            if not results_df.empty:
                results_df = results_df[results_df['Overall Score'] >= min_score]
                results_df = results_df.sort_values('Overall Score', ascending=False)
            
            return results_df
            
        except Exception as e:
            print(f"Error in pair selection analysis: {str(e)}")
            return pd.DataFrame()
    
    def get_selection_summary_data(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary data for selection results
        
        Args:
            results_df: DataFrame with analysis results
            
        Returns:
            Dictionary with summary metrics for KPI cards
        """
        if results_df.empty:
            return {
                'total_pairs': 0,
                'filtered_pairs': 0,
                'best_pair': 'N/A',
                'best_score': 0
            }
        
        total_pairs = len(results_df)
        filtered_pairs = len(results_df[results_df['Overall Score'] >= 40])
        best_pair = results_df.iloc[0] if not results_df.empty else None
        best_score = best_pair['Overall Score'] if best_pair is not None else 0
        
        return {
            'total_pairs': total_pairs,
            'filtered_pairs': filtered_pairs,
            'best_pair': best_pair['Pair'] if best_pair is not None else 'N/A',
            'best_score': best_score,
            'success_rate': f"{(filtered_pairs/total_pairs*100):.1f}%" if total_pairs > 0 else "0%",
            'score_quality': "Excellent" if best_score >= 80 else "Good" if best_score >= 60 else "Moderate"
        }
    
 # backend.py (inside class DashboardDataProvider)

    def run_trade_analysis(self, strong_asset: str, weak_asset: str, 
                        timeframe: str, data_points: int, settings: dict) -> Optional[pd.DataFrame]:
        try:
            # 1) Fetch pair data
            df = self.processor.fetch_pair_data(strong_asset, weak_asset, timeframe, data_points)
            if df is None:
                return None

            # (optional) keep these rolling stats you already compute
            df['Rolling Corr'] = df[f'LN {strong_asset} Var %'].rolling(window=30).corr(
                df[f'LN {weak_asset} Var %']
            )
            cov = df[f'LN {weak_asset} Var %'].rolling(window=30).cov(
                df[f'LN {strong_asset} Var %']
            )
            var = df[f'LN {strong_asset} Var %'].rolling(window=30).var()
            df['Rolling Beta'] = cov / var
            df['beta'] = df['Rolling Beta']

            # 2) Compose signal settings HERE (backend-only)
            signal_settings = dict(settings or {})       # safe copy even if None
            signal_settings['strong_coin'] = strong_asset
            signal_settings['weak_coin']   = weak_asset
            signal_settings['timeframe']   = timeframe

            # (Optional) gate with tactical 7D cointegration
            win7 = days_to_window(7, timeframe)
            p1 = np.log(df[f'{strong_asset} Close']).tail(win7).dropna()
            p2 = np.log(df[f'{weak_asset} Close']).tail(win7).dropna()
            c7 = self.stats.calculate_cointegration(p1, p2)
            signal_settings['cointegration_7d_ok'] = c7.get('is_cointegrated', False)

            # 3) Generate signals using your updated generator
            df = self.signal_gen.generate_signals(df, signal_settings)
            return df

        except Exception as e:
            print(f"Error in trade analysis: {str(e)}")
            return None

        
    def get_trade_analysis_summary(self, df: pd.DataFrame, strong_asset: str, weak_asset: str) -> Dict[str, Any]:
        """
        Get summary data for trade analysis
        
        Args:
            df: DataFrame with analysis results
            strong_asset: Strong asset symbol
            weak_asset: Weak asset symbol
            
        Returns:
            Dictionary with analysis summary
        """
        if df.empty or 'mad_z_score' not in df.columns:
            return {}
        
        # Current trading signal
        current_z = df['mad_z_score'].iloc[-1]
        
        # Determine signal type
        if current_z <= -2:
            signal_type = "STRONG BUY"
            signal_strength = "STRONG" if abs(current_z) > 2.5 else "MODERATE"
            signal_icon = "🟢"
            signal_action = "Open long position or add to existing"
            exit_target = "Exit when Z-score returns to 0 (median)"
        elif current_z >= 2:
            signal_type = "STRONG SELL"
            signal_strength = "STRONG" if abs(current_z) > 2.5 else "MODERATE"
            signal_icon = "🔴"
            signal_action = "Open short position or reduce long"
            exit_target = "Exit when Z-score returns to 0 (median)"
        elif -1 <= current_z <= 1:
            signal_type = "NEUTRAL"
            signal_strength = "HOLD"
            signal_icon = "⚖️"
            signal_action = "Hold or close existing positions"
            exit_target = "Wait for |Z| > 2 for new entry"
        else:
            signal_type = "WEAK SIGNAL"
            signal_strength = "MONITOR"
            signal_icon = "📊"
            signal_action = "Monitor closely, prepare for entry"
            exit_target = "Wait for stronger signal"
        
        # Distance from median
        median_spread = df['log_spread'].median()
        current_spread = df['log_spread'].iloc[-1]
        distance_from_median = ((current_spread - median_spread) / median_spread) * 100
        
        # Current correlation
        current_corr = df['Rolling Corr'].iloc[-1] if 'Rolling Corr' in df.columns else 0
        corr_strength = "Strong" if current_corr > 0.7 else "Moderate" if current_corr > 0.5 else "Weak"
        
        return {
            'current_signal': {
                'type': signal_type,
                'strength': signal_strength,
                'icon': signal_icon,
                'z_score': current_z,
                'action': signal_action,
                'exit_target': exit_target
            },
            'spread_metrics': {
                'distance_from_median': distance_from_median,
                'current_spread': current_spread,
                'median_spread': median_spread
            },
            'correlation_metrics': {
                'current_correlation': current_corr,
                'strength': corr_strength
            }
        }
    
    def get_cointegration_analysis(self, df: pd.DataFrame, strong_asset: str, weak_asset: str) -> Dict[str, Any]:
        """
        Perform dual-timeframe cointegration analysis (7D & 30D) with timeframe-aware windows.
        """
        if df.empty or len(df) < 30:
            return {}

        try:
            # 1) Resolve timeframe (no settings needed)
            tf = df.attrs.get('timeframe', '1h')  # falls back to '1h' if not set

            # 2) Convert day horizons to number of bars for this timeframe
            win7  = days_to_window(7, tf)
            win30 = days_to_window(30, tf)

            # 3) Log prices
            strong_prices = np.log(df[f'{strong_asset} Close'].dropna())
            weak_prices   = np.log(df[f'{weak_asset} Close'].dropna())

            # 4) Cointegration with timeframe-aware windows
            coint_7d  = self.stats.calculate_cointegration(strong_prices, weak_prices, window=win7)
            coint_30d = self.stats.calculate_cointegration(strong_prices, weak_prices, window=win30)

            # 5) Build status
            both_cointegrated = coint_7d['is_cointegrated'] and coint_30d['is_cointegrated']

            if both_cointegrated:
                overall_status = {
                    'status': 'Both Timeframes Cointegrated',
                    'interpretation': 'Strong mean-reverting relationship detected on both tactical and strategic timeframes - IDEAL for trading!',
                    'recommendation': 'TRADE RECOMMENDED',
                    'recommendation_detail': 'Both 7-day (tactical) and 30-day (strategic) cointegration confirm mean reversion. Use 7-day signals for entry/exit timing and 30-day for position validation.',
                    'color': '#10B981',
                    'icon': '✅'
                }
            elif coint_7d['is_cointegrated']:
                overall_status = {
                    'status': 'Only Tactical Cointegration',
                    'interpretation': 'Short-term mean reversion detected but lacks long-term stability - Use with caution',
                    'recommendation': 'SHORT-TERM ONLY',
                    'recommendation_detail': 'Only tactical cointegration detected. Consider smaller position sizes and tighter stops. Monitor for development of longer-term cointegration.',
                    'color': '#F59E0B',
                    'icon': '⚠️'
                }
            elif coint_30d['is_cointegrated']:
                overall_status = {
                    'status': 'Only Strategic Cointegration',
                    'interpretation': 'Long-term relationship exists but short-term signals may be weak - Wait for better entry',
                    'recommendation': 'WAIT FOR ENTRY',
                    'recommendation_detail': 'Strategic cointegration exists but tactical signals are weak. Wait for short-term cointegration to develop before entering positions.',
                    'color': '#3B82F6',
                    'icon': '🔍'
                }
            else:
                overall_status = {
                    'status': 'No Cointegration',
                    'interpretation': 'No significant mean-reverting relationship on either timeframe - Not recommended for trading',
                    'recommendation': 'NOT RECOMMENDED',
                    'recommendation_detail': 'No cointegration detected on either timeframe. This pair is not suitable for mean reversion strategies at this time.',
                    'color': '#EF4444',
                    'icon': '❌'
                }

            return {
                'tactical_7d': coint_7d,
                'strategic_30d': coint_30d,
                'both_cointegrated': both_cointegrated,
                'overall_status': overall_status
            }

        except Exception as e:
            print(f"Error in cointegration analysis: {str(e)}")
            return {}

    
    def get_spread_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive spread statistics
        
        Args:
            df: DataFrame with log_spread column
            
        Returns:
            Dictionary with spread statistics
        """
        if df.empty or 'log_spread' not in df.columns:
            return {}
        
        return self.stats.calculate_spread_statistics(df['log_spread'])
    
    def generate_global_report(self, collateral: str, timeframes: List[str], 
                             settings: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate comprehensive global report across multiple timeframes
        
        Args:
            collateral: Strong asset to analyze
            timeframes: List of timeframes to include
            settings: Analysis settings
            
        Returns:
            DataFrame with global analysis results
        """
        try:
            global_data = []
            
            for tf in timeframes:
                # Get timeframe display name
                timeframe_display = settings.get('timeframe_names', {}).get(tf, tf)
                
                # Run analysis for this timeframe
                results = self.analyzer.analyze_all_pairs(
                    collateral, tf,
                    settings.get('cointegration_window', 30),
                    settings.get('correlation_window', 30),
                    settings.get('beta_window', 30)
                )
                
                if not results.empty:
                    results['Timeframe'] = timeframe_display
                    global_data.append(results)
            
            if global_data:
                return pd.concat(global_data, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error generating global report: {str(e)}")
            return pd.DataFrame()
    
    def get_portfolio_data(self) -> Dict[str, Any]:
        """
        Get sample portfolio data for demonstration
        
        Returns:
            Dictionary with portfolio metrics and positions
        """
        # Sample portfolio data - in production, this would come from a database
        return {
            'summary': {
                'portfolio_value': '$24,567',
                'portfolio_delta': '+12.4%',
                'active_positions': '3/5',
                'utilization': '60% utilized',
                'todays_pnl': '+$2,456',
                'pnl_delta': '+8.2%',
                'win_rate': '73.5%',
                'win_rate_delta': '+5.2%'
            },
            'positions': pd.DataFrame({
                'Pair': ['ETH/ARB', 'ETH/ATOM', 'BTC/ALGO'],
                'Direction': ['SHORT', 'SHORT', 'LONG'],
                'Entry Price': [0.00045, 0.00123, 0.00234],
                'Current Price': [0.00041, 0.00125, 0.00238],
                'P&L %': [8.2, -1.5, 1.7],
                'P&L $': [820, -150, 170],
                'MAD Z-Score': [-2.3, 1.8, -2.1],
                'Time Held': ['2h 15m', '45m', '1h 30m'],
                'Status': ['🟢 Profitable', '🟡 Monitor', '🟢 Profitable']
            }),
            'risk_distribution': {
                'Low': {'positions': 1, 'allocation': 30},
                'Medium': {'positions': 1, 'allocation': 50},
                'High': {'positions': 1, 'allocation': 20}
            },
            'performance_metrics': {
                'Sharpe Ratio': '2.34',
                'Max Drawdown': '-8.5%',
                'Avg Win': '+4.2%',
                'Avg Loss': '-2.1%',
                'Profit Factor': '2.0'
            }
        }
    
def generate_comprehensive_excel_export(
    self,
    strong_asset: str,
    weak_asset: str,
    df: pd.DataFrame,
    analysis_summary: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Generate comprehensive Excel export with multiple sheets
    Similar to ETH_BCH_multi_interval_analysis.xlsx structure

    Args:
        strong_asset: Strong asset symbol
        weak_asset: Weak asset symbol
        df: Analysis DataFrame
        analysis_summary: Summary data from analysis

    Returns:
        Dictionary of DataFrames for Excel export (sheet_name: dataframe)
    """
    export_sheets: Dict[str, pd.DataFrame] = {}

    # Sheet 1: Raw Price Data
    price_data = df[[
        'ISO Date',
        f'{strong_asset} Close',
        f'{weak_asset} Close',
        f'LN {strong_asset} Close',
        f'LN {weak_asset} Close'
    ]].copy()
    export_sheets['Price_Data'] = price_data

    # Sheet 2: Returns & Spreads
    returns_data = df[[
        'ISO Date',
        f'LN {strong_asset} Var %',
        f'LN {weak_asset} Var %',
        'Point Spread',
        'Accum Spread',
        'log_spread'
    ]].copy()
    export_sheets['Returns_Spreads'] = returns_data

    # Sheet 3: MAD Analysis (if available)
    if 'mad_z_score' in df.columns:
        mad_data = df[[
            'ISO Date',
            'log_spread',
            'spread_median',
            'spread_mad',
            'mad_z_score'
        ]].copy()

        # Add MAD thresholds
        mad_data['MAD_Upper_2'] = mad_data['spread_median'] + 2 * mad_data['spread_mad'] * 1.4826
        mad_data['MAD_Lower_2'] = mad_data['spread_median'] - 2 * mad_data['spread_mad'] * 1.4826
        mad_data['MAD_Upper_3'] = mad_data['spread_median'] + 3 * mad_data['spread_mad'] * 1.4826
        mad_data['MAD_Lower_3'] = mad_data['spread_median'] - 3 * mad_data['spread_mad'] * 1.4826

        export_sheets['MAD_Analysis'] = mad_data

    # Sheet 4: Trading Signals (if available)
    if 'entry_signal' in df.columns or 'exit_signal' in df.columns:
        base_cols = ['ISO Date']
        if 'log_spread' in df.columns:
            base_cols.append('log_spread')
        if 'mad_z_score' in df.columns:
            base_cols.append('mad_z_score')

        signal_data = df[base_cols].copy()

        if 'entry_signal' in df.columns:
            signal_data['Entry_Signal'] = df['entry_signal']
        if 'exit_signal' in df.columns:
            signal_data['Exit_Signal'] = df['exit_signal']

        # Add signal interpretation (if z is present)
        if 'mad_z_score' in signal_data.columns:
            signal_data['Signal_Type'] = signal_data['mad_z_score'].apply(
                lambda z: 'STRONG_BUY' if z <= -2 else 'STRONG_SELL' if z >= 2 else 'NEUTRAL'
            )

        export_sheets['Trading_Signals'] = signal_data

    # Sheet 5: Rolling Statistics (if available)
    if 'Rolling Corr' in df.columns:
        rolling_cols = ['ISO Date', 'Rolling Corr', 'Rolling Beta']
        rolling_stats = df[[c for c in rolling_cols if c in df.columns]].copy()

        # Add rolling statistics based on spread, if available
        if 'log_spread' in df.columns:
            rolling_stats['Rolling_Mean'] = df['log_spread'].rolling(30).mean()
            rolling_stats['Rolling_Std'] = df['log_spread'].rolling(30).std()
            rolling_stats['Rolling_MAD'] = df['log_spread'].rolling(30).apply(
                lambda x: np.median(np.abs(x - x.median()))
            )

        export_sheets['Rolling_Statistics'] = rolling_stats

    # Sheet 6: Summary Statistics
    if 'log_spread' in df.columns:
        spread_stats = self.get_spread_statistics(df)

        summary_df = pd.DataFrame([{
            'Metric': 'Pair',
            'Value': f'{strong_asset}/{weak_asset}',
            'Description': 'Trading pair analyzed'
        }, {
            'Metric': 'Data Points',
            'Value': len(df),
            'Description': 'Number of observations'
        }, {
            'Metric': 'Mean Spread',
            'Value': spread_stats.get('mean', 0),
            'Description': 'Average log spread'
        }, {
            'Metric': 'Median Spread',
            'Value': spread_stats.get('median', 0),
            'Description': 'Median log spread (mean reversion target)'
        }, {
            'Metric': 'MAD',
            'Value': spread_stats.get('mad', 0),
            'Description': 'Median Absolute Deviation'
        }, {
            'Metric': 'Standard Deviation',
            'Value': spread_stats.get('std', 0),
            'Description': 'Standard deviation of spread'
        }])

        export_sheets['Summary_Statistics'] = summary_df

    # Sheet 7: Cointegration Analysis (if available)
    if analysis_summary and 'tactical_7d' in analysis_summary:
        coint_data = []

        # 7-day tactical
        tactical = analysis_summary['tactical_7d']
        coint_data.append({
            'Timeframe': '7-Day Tactical',
            'Window': 7,
            'P_Value': tactical.get('pvalue', 1.0),
            'Is_Cointegrated': tactical.get('is_cointegrated', False),
            'Confidence': tactical.get('confidence', 'Failed'),
            'Purpose': 'Entry/Exit Signals'
        })

        # 30-day strategic
        if 'strategic_30d' in analysis_summary:
            strategic = analysis_summary['strategic_30d']
            coint_data.append({
                'Timeframe': '30-Day Strategic',
                'Window': 30,
                'P_Value': strategic.get('pvalue', 1.0),
                'Is_Cointegrated': strategic.get('is_cointegrated', False),
                'Confidence': strategic.get('confidence', 'Failed'),
                'Purpose': 'Pair Validation'
            })

        export_sheets['Cointegration_Analysis'] = pd.DataFrame(coint_data)

    # Sheet 8: Trading Recommendations
    if analysis_summary and 'overall_status' in analysis_summary:
        recommendations = pd.DataFrame([{
            'Recommendation': analysis_summary['overall_status'].get('recommendation', 'N/A'),
            'Status': analysis_summary['overall_status'].get('status', 'N/A'),
            'Interpretation': analysis_summary['overall_status'].get('interpretation', 'N/A'),
            'Action_Plan': analysis_summary['overall_status'].get('recommendation_detail', 'N/A'),
            'Risk_Level': 'Low' if analysis_summary.get('both_cointegrated', False) else 'High',
            'Confidence': 'High' if analysis_summary.get('both_cointegrated', False) else 'Low'
        }])

        export_sheets['Trading_Recommendations'] = recommendations

    return export_sheets
