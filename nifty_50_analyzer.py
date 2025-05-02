# -*- coding: utf-8 -*-
"""
Nifty 50 Stock Analysis using StockPrediction.py
This script runs the StockModelComparison class from StockPrediction.py on all Nifty 50 stocks
and identifies the best performers for the next 5-day period.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import the StockModelComparison class from StockPrediction.py
from StockPrediction import StockModelComparison

class Nifty50Analyzer:
    def __init__(self, test_period_months=1, initial_investment=100000, start_date='2020-01-01'):
        """Initialize the analyzer"""
        self.test_period_months = test_period_months
        self.initial_investment = initial_investment
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Nifty 50 stocks list
        self.nifty50_stocks = [ 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'  , 'INFY.NS', 
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS',
            'LICI.NS', 'KOTAKBANK.NS', 'HCLTECH.NS', 'AXISBANK.NS', 'LARSEN.NS', 
            'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'WIPRO.NS',
            'ONGC.NS', 'NTPC.NS', 'TATAMOTORS.NS', 'BAJAJFINSV.NS', 'ADANIENT.NS',
            'ULTRACEMCO.NS', 'JSWSTEEL.NS', 'POWERGRID.NS', 'DIVISLAB.NS', 'ADANIPORTS.NS',
            'NESTLEIND.NS', 'TATACONSUM.NS', 'TATASTEEL.NS', 'INDUSINDBK.NS', 'M&M.NS',
            'DRREDDY.NS', 'GRASIM.NS', 'TECHM.NS', 'HDFCLIFE.NS', 'APOLLOHOSP.NS',
            'COALINDIA.NS', 'HINDALCO.NS', 'SBILIFE.NS', 'CIPLA.NS', 'BAJAJ-AUTO.NS',
            'BRITANNIA.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'UPL.NS', 'BPCL.NS'
        ]
        
        
        # Store results
        self.results = {}
        
    def analyze_stock(self, ticker):
        """Analyze a single stock using StockModelComparison"""
        print(f"Analyzing {ticker}...")
        
        try:
            # Create StockModelComparison instance
            stock_model = StockModelComparison(
                ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                test_period_months=self.test_period_months,
                initial_investment=self.initial_investment
            )
            
            # Configure the analysis for 5-day prediction
            stock_model.set_target_type("4")  # 5-day direction
            
            # Run complete analysis pipeline
            stock_model.download_data()
            stock_model.calculate_technical_indicators()
            stock_model.prepare_features()
            stock_model.build_models()
            stock_model.train_evaluate_models()
            
            # Generate trading signals
            stock_model.generate_trading_signals()
            
            # Get test period performance
            test_performance = stock_model.estimate_test_returns()
            
            # Get last price and signal
            last_price = stock_model.stock_data['Close'].iloc[-1]
            latest_signal = stock_model.stock_data['ML_Signal'].iloc[-1] if 'ML_Signal' in stock_model.stock_data.columns else None
            
            # Store results
            return {
                'ticker': ticker,
                'best_model': stock_model.best_model_name,
                'test_return': test_performance['Strategy Return'] if test_performance else None,
                'market_return': test_performance['Market Return'] if test_performance else None,
                'alpha': test_performance['Alpha'] if test_performance else None,
                'sharpe': test_performance['Strategy Sharpe'] if test_performance else None,
                'win_rate': test_performance['Win Rate'] if test_performance else None,
                'profit_factor': test_performance['Profit Factor'] if test_performance else None,
                'last_price': last_price,
                'latest_signal': latest_signal,
                'potential_5day_return': test_performance['Strategy Annual Return'] / 50 if test_performance else None,  # Approximating 5-day return
                'recommendation': 'BUY' if latest_signal == 1 else 'SELL/HOLD' if latest_signal == 0 else 'UNKNOWN'
            }
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e)
            }
    
    def run_analysis(self, max_workers=4):
        """Run analysis on all stocks with parallel processing"""
        print(f"🚀 Starting analysis on {len(self.nifty50_stocks)} Nifty 50 stocks...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(self.analyze_stock, ticker): ticker for ticker in self.nifty50_stocks}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if 'error' not in result:
                        self.results[ticker] = result
                        print(f"✅ {ticker} analysis complete")
                    else:
                        print(f"❌ {ticker} analysis failed: {result['error']}")
                except Exception as e:
                    print(f"❌ {ticker} analysis failed with exception: {str(e)}")
        
        return self.get_recommendations()
    
    def get_recommendations(self):
        """Analyze results and recommend stocks"""
        if not self.results:
            print("No valid results available.")
            return None
        
        # Convert to DataFrame
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        
        # Clean DataFrame
        valid_results = results_df.dropna(subset=['test_return', 'alpha', 'sharpe'])
        
        # Get BUY recommendations
        buy_recommendations = valid_results[valid_results['recommendation'] == 'BUY']
        
        # Calculate ranks for each metric (handle empty dataframes gracefully)
        if not valid_results.empty:
            # Create composite score - making sure to handle the possibility of missing values
            try:
                # First compute individual rankings
                alpha_rank = valid_results['alpha'].rank(pct=True)
                return_rank = valid_results['test_return'].rank(pct=True)
                sharpe_rank = valid_results['sharpe'].rank(pct=True)
                win_rate_rank = valid_results['win_rate'].rank(pct=True)
                pf_rank = valid_results['profit_factor'].rank(pct=True)
                
                # Then create the composite score
                valid_results['composite_score'] = (
                    alpha_rank * 0.3 +
                    return_rank * 0.25 +
                    sharpe_rank * 0.2 +
                    win_rate_rank * 0.15 +
                    pf_rank * 0.1
                )
            except Exception as e:
                print(f"Error calculating composite score: {str(e)}")
                valid_results['composite_score'] = valid_results['test_return'].rank(pct=True)  # Fallback to just using returns
        
        # Sort by different metrics
        top_alpha = valid_results.sort_values('alpha', ascending=False).head(10) if 'alpha' in valid_results.columns else pd.DataFrame()
        top_returns = valid_results.sort_values('test_return', ascending=False).head(10) if 'test_return' in valid_results.columns else pd.DataFrame()
        top_sharpe = valid_results.sort_values('sharpe', ascending=False).head(10) if 'sharpe' in valid_results.columns else pd.DataFrame()
        top_overall = valid_results.sort_values('composite_score', ascending=False).head(10) if 'composite_score' in valid_results.columns else pd.DataFrame()
        
        # Handle buy recommendations (or lack thereof)
        if not buy_recommendations.empty and 'composite_score' in buy_recommendations.columns:
            top_buy = buy_recommendations.sort_values('composite_score', ascending=False).head(5)
        else:
            print("Warning: No BUY recommendations available. Using top overall performers instead.")
            top_buy = top_overall.head(5) if not top_overall.empty else pd.DataFrame()
        
        # Plot the comparison of top 5 stocks
        if not top_buy.empty and len(top_buy) >= 3:  # Only plot if we have at least 3 stocks
            plt.figure(figsize=(16, 12))
            
            # Plot test returns
            plt.subplot(2, 2, 1)
            sns.barplot(x=top_buy.index, y='test_return', data=top_buy)
            plt.title('Test Period Returns (%)')
            plt.xticks(rotation=45)
            plt.ylabel('Return (%)')
            
            # Plot alpha
            plt.subplot(2, 2, 2)
            sns.barplot(x=top_buy.index, y='alpha', data=top_buy)
            plt.title('Alpha')
            plt.xticks(rotation=45)
            plt.ylabel('Alpha')
            
            # Plot Sharpe ratio
            plt.subplot(2, 2, 3)
            sns.barplot(x=top_buy.index, y='sharpe', data=top_buy)
            plt.title('Sharpe Ratio')
            plt.xticks(rotation=45)
            plt.ylabel('Sharpe Ratio')
            
            # Plot win rate
            plt.subplot(2, 2, 4)
            sns.barplot(x=top_buy.index, y='win_rate', data=top_buy)
            plt.title('Win Rate')
            plt.xticks(rotation=45)
            plt.ylabel('Win Rate (%)')
            
            plt.tight_layout()
            try:
                plt.savefig('top_stocks_comparison.png')  # Save figure for reference
            except Exception as e:
                print(f"Warning: Could not save figure: {str(e)}")
            plt.show()
            
            # Plot expected 5-day returns for top picks
            plt.figure(figsize=(10, 6))
            returns_data = top_buy['potential_5day_return'] * 100
            sns.barplot(x=top_buy.index, y=returns_data)
            plt.title('Expected 5-Day Returns for Top Picks (%)')
            plt.xticks(rotation=45)
            plt.ylabel('Expected Return (%)')
            plt.tight_layout()
            try:
                plt.savefig('expected_returns.png')  # Save figure for reference
            except Exception as e:
                print(f"Warning: Could not save figure: {str(e)}")
            plt.show()
        
        return {
            'all_results': valid_results,
            'top_alpha': top_alpha,
            'top_returns': top_returns,
            'top_sharpe': top_sharpe,
            'top_overall': top_overall,
            'top_buy': top_buy
        }
    
    def display_results(self, recommendations):
        """Display analysis results in a clear format"""
        if not recommendations:
            print("No valid recommendations available.")
            return
        
        print("\n" + "=" * 80)
        print("🏆 TOP NIFTY 50 STOCKS TO BUY FOR NEXT 5 DAYS")
        print("=" * 80)
        
        top_buy = recommendations['top_buy']
        
        if top_buy.empty:
            print("No buy recommendations available based on the analysis.")
            return
        
        for i, (ticker, row) in enumerate(top_buy.iterrows(), 1):
            company_name = ticker.split('.')[0]
            print(f"{i}. {company_name}")
            print(f"   Last Price: ₹{row['last_price']:.2f}")
            print(f"   Test Period Return: {row['test_return']*100:.2f}%")
            print(f"   Expected 5-Day Return: {row['potential_5day_return']*100:.2f}%")
            print(f"   Alpha: {row['alpha']*100:.2f}%")
            print(f"   Sharpe Ratio: {row['sharpe']:.2f}")
            print(f"   Win Rate: {row['win_rate']*100:.2f}%")
            print(f"   Best Model: {row['best_model']}")
            print()
        
        print("\n" + "=" * 80)
        print("📊 INVESTMENT ALLOCATION FOR ₹{:,.2f}".format(self.initial_investment))
        print("=" * 80)
        
        if not top_buy.empty:
            # Calculate allocation
            investment_per_stock = self.initial_investment / len(top_buy)
            
            total_actual_investment = 0
            total_expected_return = 0
            
            for ticker, row in top_buy.iterrows():
                company_name = ticker.split('.')[0]
                shares = int(investment_per_stock / row['last_price'])
                actual_investment = shares * row['last_price']
                expected_gain = actual_investment * row['potential_5day_return']
                
                print(f"{company_name}: Buy {shares} shares at ₹{row['last_price']:.2f} = ₹{actual_investment:,.2f}")
                print(f"Expected 5-day gain: ₹{expected_gain:,.2f} ({row['potential_5day_return']*100:.2f}%)")
                print()
                
                total_actual_investment += actual_investment
                total_expected_return += expected_gain
            
            print("-" * 80)
            print(f"Total Investment: ₹{total_actual_investment:,.2f}")
            print(f"Total Expected Gain: ₹{total_expected_return:,.2f} ({total_expected_return/total_actual_investment*100:.2f}%)")
            
            # Calculate remaining cash
            remaining_cash = self.initial_investment - total_actual_investment
            print(f"Remaining Cash: ₹{remaining_cash:,.2f}")
        
        print("\n" + "=" * 80)
        print("📈 OVERALL NIFTY 50 PERFORMANCE RANKING (TOP 10)")
        print("=" * 80)
        
        top_overall = recommendations['top_overall']
        
        if not top_overall.empty:
            # Get relevant columns, handling the case where some might not exist
            cols_to_display = ['last_price', 'test_return', 'alpha', 'sharpe', 'win_rate', 'recommendation']
            available_cols = [col for col in cols_to_display if col in top_overall.columns]
            
            if available_cols:
                display_df = top_overall[available_cols]
                
                # Format percentages for display
                formatted_overall = display_df.copy()
                if 'test_return' in formatted_overall.columns:
                    formatted_overall['test_return'] = formatted_overall['test_return'].apply(lambda x: f"{x*100:.2f}%")
                if 'alpha' in formatted_overall.columns:
                    formatted_overall['alpha'] = formatted_overall['alpha'].apply(lambda x: f"{x*100:.2f}%")
                if 'win_rate' in formatted_overall.columns:
                    formatted_overall['win_rate'] = formatted_overall['win_rate'].apply(lambda x: f"{x*100:.2f}%")
                
                # Display with better column names
                col_mapping = {
                    'last_price': 'Price (₹)', 
                    'test_return': 'Test Return', 
                    'alpha': 'Alpha', 
                    'sharpe': 'Sharpe', 
                    'win_rate': 'Win Rate', 
                    'recommendation': 'Signal'
                }
                formatted_overall.columns = [col_mapping.get(col, col) for col in formatted_overall.columns]
                formatted_overall.index = [idx.split('.')[0] for idx in formatted_overall.index]
                
                print(formatted_overall)
            else:
                print("No data available to display.")
        else:
            print("No overall performance data available.")
        
        return top_buy


if __name__ == "__main__":
    print("=" * 80)
    print("🔍 NIFTY 50 STOCK ANALYSIS FOR NEXT 5 DAYS")
    print("=" * 80)
    
    # Get user inputs
    start_date = input("Enter start date for historical data (YYYY-MM-DD) (default: 2020-01-01): ") or "2020-01-01"
    
    try:
        test_period = int(input("Enter test period in months (default: 1): ") or "1")
    except ValueError:
        print("Invalid input, using default 1 month")
        test_period = 1
    
    try:
        initial_investment = float(input("Enter initial investment amount in ₹ (default: 100000): ") or "100000")
    except ValueError:
        print("Invalid input, using default ₹100,000")
        initial_investment = 100000
    
    try:
        max_workers = int(input("Enter number of parallel processes (default: 4): ") or "4")
    except ValueError:
        print("Invalid input, using default 4 processes")
        max_workers = 4
    
    # Run analysis
    analyzer = Nifty50Analyzer(
        test_period_months=test_period,
        initial_investment=initial_investment,
        start_date=start_date
    )
    
    # Display configuration
    print("\n" + "=" * 80)
    print(f"📊 ANALYSIS CONFIGURATION:")
    print(f"📅 Historical Data: {start_date} to {analyzer.end_date}")
    print(f"🧪 Test Period: {test_period} months")
    print(f"💰 Investment Amount: ₹{initial_investment:,.2f}")
    print(f"⚙️ Parallel Processes: {max_workers}")
    print("=" * 80)
    
    # Run the analysis
    print("\n🚀 Starting analysis on Nifty 50 stocks...")
    recommendations = analyzer.run_analysis(max_workers=max_workers)
    
    # Display results
    top_picks = analyzer.display_results(recommendations)