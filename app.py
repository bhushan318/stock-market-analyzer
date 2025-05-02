# -*- coding: utf-8 -*-
"""
Created on Fri May  2 00:51:42 2025

@author: NagabhushanamTattaga
"""

import streamlit as st
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import required modules with fallbacks
try:
    from StockPrediction import StockModelComparison
    from nifty_50_analyzer import Nifty50Analyzer
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.info("Some dependencies might be missing. Please check the requirements.txt file.")
    import sys
    st.write(f"Python version: {sys.version}")
    st.stop()


# Set page configuration
st.set_page_config(
    page_title="Stock Market Analyzer",
    page_icon="📈",
    layout="wide"
)





def analyze_portfolio_holdings(portfolio_df, start_date, end_date, test_period, model_choice, indicator_choice):
    """
    Analyze each stock in the portfolio individually to generate recommendations
    
    Returns:
    - analysis_results: DataFrame with analysis results for each stock
    - portfolio_metrics: Overall portfolio metrics
    - recommendations: Recommendations for portfolio optimization
    """
    
    analysis_results = {}
    
    total_investment = portfolio_df['Investment'].sum()
    total_current_value = 0
    total_expected_value = 0
    
    progress_bar = st.progress(0, text="Starting portfolio analysis...")
    
    # Analyze each stock in the portfolio
    for idx, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        shares = row['Shares']
        avg_price = row['Avg Price']
        investment = row['Investment']
        
        progress_bar.progress((idx + 1) / len(portfolio_df), text=f"Analyzing {ticker}...")
        
        try:
            # Use StockModelComparison for detailed analysis
            stock_model = StockModelComparison(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                test_period_months=test_period,
                initial_investment=investment
            )
            
            # Configure based on user preferences
            stock_model.set_enabled_models(model_choice)
            stock_model.set_indicator_level(indicator_choice)
            stock_model.set_target_type("1")  # Next day direction for portfolio analysis
            
            # Run the complete analysis pipeline
            stock_model.download_data()
            stock_model.calculate_technical_indicators()
            stock_model.prepare_features()
            stock_model.build_models()
            stock_model.train_evaluate_models()
            stock_model.generate_trading_signals()
            
            # Get performance metrics from test period
            test_metrics = stock_model.estimate_test_returns()
            
            # Get last price and signal
            last_price = stock_model.stock_data['Close'].iloc[-1]
            latest_signal = stock_model.stock_data['ML_Signal'].iloc[-1] if 'ML_Signal' in stock_model.stock_data.columns else None
            
            # Signal strength analysis
            signal_confidence = "Medium"
            if latest_signal == 1 and stock_model.stock_data['ML_Signal'].iloc[-5:].mean() > 0.8:
                signal_confidence = "Strong Buy"
            elif latest_signal == 0 and stock_model.stock_data['ML_Signal'].iloc[-5:].mean() < 0.2:
                signal_confidence = "Strong Sell"
            
            # Calculate current value and P&L
            current_value = shares * last_price
            pnl = current_value - investment
            pnl_percent = (pnl / investment) * 100
            
            # Project expected return over next month
            expected_monthly_return = test_metrics.get('Strategy Annual Return', 0) / 12
            expected_value_1m = current_value * (1 + expected_monthly_return)
            expected_gain_1m = expected_value_1m - current_value
            
            # Calculate dynamic month forecast based on recent performance and signals
            recent_performance = stock_model.stock_data['Returns'].iloc[-30:].mean() * 20  # Scale up for monthly
            signal_factor = 1.2 if latest_signal == 1 else 0.8 if latest_signal == 0 else 1.0
            dynamic_forecast = recent_performance * signal_factor
            
            # Get best model and performance metrics
            best_model = stock_model.best_model_name
            sharpe = test_metrics.get('Strategy Sharpe', 0)
            win_rate = test_metrics.get('Win Rate', 0)
            
            # Store results
            analysis_results[ticker] = {
                'Ticker': ticker,
                'Shares': shares,
                'Avg Price': avg_price,
                'Investment': investment,
                'Current Price': last_price,
                'Current Value': current_value,
                'P&L': pnl,
                'P&L %': pnl_percent,
                'Signal': 'BUY' if latest_signal == 1 else 'SELL' if latest_signal == 0 else 'HOLD' if latest_signal == 0.5 else 'UNKNOWN',
                'Signal Confidence': signal_confidence,
                'Best Model': best_model,
                'Sharpe Ratio': sharpe,
                'Win Rate': win_rate * 100,
                'Expected Monthly Return': expected_monthly_return * 100,
                'Expected Value in 1M': expected_value_1m,
                'Expected Gain in 1M': expected_gain_1m,
                'Dynamic Forecast': dynamic_forecast * 100
            }
            
            total_current_value += current_value
            total_expected_value += expected_value_1m
            
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            analysis_results[ticker] = {
                'Ticker': ticker,
                'Shares': shares,
                'Avg Price': avg_price,
                'Investment': investment,
                'Error': str(e)
            }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(analysis_results, orient='index')
    
    # Calculate portfolio metrics
    total_pnl = total_current_value - total_investment
    total_pnl_percent = (total_pnl / total_investment) * 100
    total_expected_gain = total_expected_value - total_current_value
    total_expected_return_percent = (total_expected_gain / total_current_value) * 100
    
    portfolio_metrics = {
        'total_investment': total_investment,
        'total_current_value': total_current_value,
        'total_pnl': total_pnl,
        'total_pnl_percent': total_pnl_percent,
        'total_expected_value': total_expected_value,
        'total_expected_gain': total_expected_gain,
        'total_expected_return_percent': total_expected_return_percent
    }
    
    # Generate portfolio recommendations
    recommendations = generate_portfolio_recommendations(results_df, portfolio_metrics)
    
    progress_bar.empty()
    
    return results_df, portfolio_metrics, recommendations

def generate_portfolio_recommendations(results_df, portfolio_metrics):
    """
    Generate portfolio recommendations based on stock analysis
    """
    recommendations = {
        'buy_recommendations': [],
        'sell_recommendations': [],
        'hold_recommendations': [],
        'rebalance_suggestions': [],
        'overall_strategy': '',
        'sector_diversification': '',
        'risk_analysis': '',
        'expected_performance': ''
    }
    
    if 'Signal' not in results_df.columns:
        return recommendations
    
    # Extract recommendations based on signals
    buy_signals = results_df[results_df['Signal'] == 'BUY']
    sell_signals = results_df[results_df['Signal'] == 'SELL']
    hold_signals = results_df[results_df['Signal'] == 'HOLD']
    
    # Process buy recommendations
    for ticker, row in buy_signals.iterrows():
        confidence = row.get('Signal Confidence', 'Medium')
        expected_return = row.get('Expected Monthly Return', 0)
        sharpe = row.get('Sharpe Ratio', 0)
        
        strength = "Strong" if confidence == "Strong Buy" or expected_return > 5 or sharpe > 1.5 else "Moderate"
        
        recommendations['buy_recommendations'].append({
            'ticker': ticker,
            'current_price': row['Current Price'],
            'expected_return': expected_return,
            'strength': strength,
            'rationale': f"Expected {expected_return:.2f}% return in next month with {sharpe:.2f} Sharpe ratio"
        })
    
    # Process sell recommendations
    for ticker, row in sell_signals.iterrows():
        confidence = row.get('Signal Confidence', 'Medium')
        expected_return = row.get('Expected Monthly Return', 0)
        
        strength = "Strong" if confidence == "Strong Sell" or expected_return < -2 else "Moderate"
        
        recommendations['sell_recommendations'].append({
            'ticker': ticker,
            'current_price': row['Current Price'],
            'expected_return': expected_return,
            'strength': strength,
            'rationale': f"Expected {expected_return:.2f}% return in next month. Consider reducing position."
        })
    
    # Process hold recommendations
    for ticker, row in hold_signals.iterrows():
        expected_return = row.get('Expected Monthly Return', 0)
        
        recommendations['hold_recommendations'].append({
            'ticker': ticker,
            'current_price': row['Current Price'],
            'expected_return': expected_return,
            'rationale': f"Holding with expected {expected_return:.2f}% return in next month."
        })
    
    # Generate rebalancing suggestions
    if len(results_df) > 0:
        # Calculate ideal weights
        total_value = results_df['Current Value'].sum()
        
        # Prioritize stocks with positive expected returns
        positive_return_stocks = results_df[results_df['Expected Monthly Return'] > 0]
        
        # Calculate target allocation based on expected returns and signals
        for ticker, row in results_df.iterrows():
            current_weight = row['Current Value'] / total_value
            
            if row['Signal'] == 'BUY':
                # For buy signals, increase allocation
                target_weight = min(current_weight * 1.5, 0.30)  # Cap at 30% for diversification
            elif row['Signal'] == 'SELL':
                # For sell signals, decrease allocation
                target_weight = max(current_weight * 0.5, 0.05)  # Minimum 5% to maintain position
            else:
                # For hold signals, maintain similar allocation
                target_weight = current_weight
            
            # Adjust based on expected return
            expected_return = row.get('Expected Monthly Return', 0)
            if expected_return > 5:
                target_weight *= 1.2
            elif expected_return < -2:
                target_weight *= 0.8
            
            target_value = target_weight * total_value
            current_value = row['Current Value']
            change_value = target_value - current_value
            
            # Calculate share change based on current price
            change_shares = int(change_value / row['Current Price']) if row['Current Price'] > 0 else 0
            
            action = 'BUY' if change_shares > 0 else 'SELL' if change_shares < 0 else 'HOLD'
            
            recommendations['rebalance_suggestions'].append({
                'ticker': ticker,
                'current_shares': row['Shares'],
                'current_weight': current_weight * 100,
                'target_weight': target_weight * 100,
                'change_shares': change_shares,
                'action': action,
                'rationale': f"{'Increase' if action == 'BUY' else 'Decrease' if action == 'SELL' else 'Maintain'} allocation based on expected {expected_return:.2f}% return"
            })
    
    # Overall portfolio strategy
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    hold_count = len(hold_signals)
    
    if buy_count > sell_count and portfolio_metrics['total_expected_return_percent'] > 2:
        recommendations['overall_strategy'] = "BULLISH: Portfolio shows positive momentum. Consider increasing equity exposure."
    elif sell_count > buy_count and portfolio_metrics['total_expected_return_percent'] < 0:
        recommendations['overall_strategy'] = "BEARISH: Multiple sell signals detected. Consider reducing risk exposure."
    else:
        recommendations['overall_strategy'] = "NEUTRAL: Mixed signals. Maintain current allocation with minor adjustments."
    
    # Expected performance projection
    recommendations['expected_performance'] = f"Expected portfolio return over next month: {portfolio_metrics['total_expected_return_percent']:.2f}%"
    
    return recommendations

def display_portfolio_analysis_and_recommendations(results_df, portfolio_metrics, recommendations, currency_symbol):
    """
    Display portfolio analysis results and recommendations
    """
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Display portfolio summary
    st.subheader("📊 Portfolio Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Investment", f"{currency_symbol}{portfolio_metrics['total_investment']:,.2f}")
    with col2:
        st.metric("Current Value", f"{currency_symbol}{portfolio_metrics['total_current_value']:,.2f}")
    with col3:
        st.metric("Total P&L", 
                f"{currency_symbol}{portfolio_metrics['total_pnl']:,.2f} ({portfolio_metrics['total_pnl_percent']:.2f}%)", 
                delta=f"{portfolio_metrics['total_pnl_percent']:.2f}%")
    
    # Display expected returns
    st.subheader("📈 Expected Performance (Next Month)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Value", f"{currency_symbol}{portfolio_metrics['total_current_value']:,.2f}")
    with col2:
        st.metric("Expected Value in 1M", f"{currency_symbol}{portfolio_metrics['total_expected_value']:,.2f}")
    with col3:
        st.metric("Expected Gain", 
                f"{currency_symbol}{portfolio_metrics['total_expected_gain']:,.2f} ({portfolio_metrics['total_expected_return_percent']:.2f}%)", 
                delta=f"{portfolio_metrics['total_expected_return_percent']:.2f}%")
    
    # Display stock analysis results
    st.subheader("📊 Portfolio Stock Analysis")
    
    if not results_df.empty and 'Current Value' in results_df.columns:
        # Format the DataFrame for display
        display_df = results_df.copy()
        
        # Select columns for display
        display_cols = ['Shares', 'Avg Price', 'Current Price', 'P&L %', 'Signal', 'Signal Confidence', 
                         'Expected Monthly Return', 'Sharpe Ratio']
        
        if all(col in display_df.columns for col in display_cols):
            display_df = display_df[display_cols]
            
            # Format monetary values
            for col in ['Avg Price', 'Current Price']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{currency_symbol}{x:,.2f}")
            
            # Format percentages
            for col in ['P&L %', 'Expected Monthly Return']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
            
            # Format Sharpe
            if 'Sharpe Ratio' in display_df.columns:
                display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
            
            # Add color coding for signals
            def color_signal(val):
                if val == 'BUY':
                    return 'background-color: #c6efce; color: #006100'
                elif val == 'SELL':
                    return 'background-color: #ffc7ce; color: #9c0006'
                else:
                    return 'background-color: #ffeb9c; color: #9c6500'
            
            # Display with styling
            st.dataframe(display_df.style.applymap(color_signal, subset=['Signal']), use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
    
    # Display detailed recommendations
    st.subheader("💰 Portfolio Recommendations")
    
    # Overall strategy
    st.markdown(f"### Overall Strategy")
    st.info(recommendations['overall_strategy'])
    st.markdown(f"### Expected Performance")
    st.success(recommendations['expected_performance'])
    
    # Buy recommendations
    if recommendations['buy_recommendations']:
        st.markdown("### 🛒 Recommended Buys")
        for rec in recommendations['buy_recommendations']:
            st.markdown(f"""
            **{rec['ticker']}** - {rec['strength']} Buy at {currency_symbol}{rec['current_price']:.2f}
            - Expected Return: {rec['expected_return']:.2f}%
            - Rationale: {rec['rationale']}
            """)
    
    # Sell recommendations
    if recommendations['sell_recommendations']:
        st.markdown("### 💸 Recommended Sells")
        for rec in recommendations['sell_recommendations']:
            st.markdown(f"""
            **{rec['ticker']}** - {rec['strength']} Sell at {currency_symbol}{rec['current_price']:.2f}
            - Expected Return: {rec['expected_return']:.2f}%
            - Rationale: {rec['rationale']}
            """)
    
    # Hold recommendations
    if recommendations['hold_recommendations']:
        st.markdown("### ✋ Recommended Holds")
        for rec in recommendations['hold_recommendations']:
            st.markdown(f"""
            **{rec['ticker']}** - Hold at {currency_symbol}{rec['current_price']:.2f}
            - Expected Return: {rec['expected_return']:.2f}%
            - Rationale: {rec['rationale']}
            """)
    
    # Visualize portfolio allocation
    st.subheader("📊 Portfolio Allocation")
    
    if 'Current Value' in results_df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Current allocation
        current_allocation = results_df['Current Value'].copy()
        current_allocation.index = [idx.split('.')[0] for idx in current_allocation.index]
        ax1.pie(current_allocation, labels=current_allocation.index, autopct='%1.1f%%')
        ax1.set_title('Current Allocation')
        
        # Target allocation
        if recommendations['rebalance_suggestions']:
            rebalance_df = pd.DataFrame(recommendations['rebalance_suggestions'])
            target_weights = rebalance_df['target_weight'].copy()
            target_weights.index = [rec['ticker'].split('.')[0] for rec in recommendations['rebalance_suggestions']]
            ax2.pie(target_weights, labels=target_weights.index, autopct='%1.1f%%')
            ax2.set_title('Recommended Allocation')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Visualize expected returns
    st.subheader("📈 Expected Monthly Returns by Stock")
    
    if 'Expected Monthly Return' in results_df.columns:
        # Format tickers for display
        formatted_tickers = [ticker.split('.')[0] for ticker in results_df.index]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(formatted_tickers, results_df['Expected Monthly Return'])
        
        # Color bars by signal
        for i, bar in enumerate(bars):
            ticker = results_df.index[i]
            if results_df.loc[ticker, 'Signal'] == 'BUY':
                bar.set_color('green')
            elif results_df.loc[ticker, 'Signal'] == 'SELL':
                bar.set_color('red')
            else:
                bar.set_color('gold')
        
        ax.set_ylabel('Expected Monthly Return (%)')
        ax.set_xlabel('Stock')
        ax.set_title('Expected Monthly Returns with Trading Signals')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add portfolio average line
        avg_return = portfolio_metrics['total_expected_return_percent']
        ax.axhline(y=avg_return, color='blue', linestyle='--', alpha=0.8, label=f'Portfolio Avg: {avg_return:.2f}%')
        
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    # Show rebalancing recommendations
    st.subheader("📊 Portfolio Rebalancing Plan")
    
    if recommendations['rebalance_suggestions']:
        rebalance_df = pd.DataFrame(recommendations['rebalance_suggestions'])
        
        # Format for display
        display_rebalance = rebalance_df.copy()
        display_rebalance = display_rebalance[['ticker', 'current_shares', 'current_weight', 'target_weight', 'change_shares', 'action']]
        display_rebalance.columns = ['Stock', 'Current Shares', 'Current Weight (%)', 'Target Weight (%)', 'Change Shares', 'Action']
        
        # Format percentages
        display_rebalance['Current Weight (%)'] = display_rebalance['Current Weight (%)'].apply(lambda x: f"{x:.2f}%")
        display_rebalance['Target Weight (%)'] = display_rebalance['Target Weight (%)'].apply(lambda x: f"{x:.2f}%")
        
        # Set index to ticker for better display
        display_rebalance = display_rebalance.set_index('Stock')
        
        # Display with styling
        def color_action(val):
            if val == 'BUY':
                return 'background-color: #c6efce; color: #006100'
            elif val == 'SELL':
                return 'background-color: #ffc7ce; color: #9c0006'
            else:
                return 'background-color: #ffeb9c; color: #9c6500'
        
        st.dataframe(display_rebalance.style.applymap(color_action, subset=['Action']), use_container_width=True)
        
        # Show action plan
        st.subheader("📝 Action Plan")
        
        # Buy actions
        buy_actions = [rec for rec in recommendations['rebalance_suggestions'] if rec['action'] == 'BUY']
        if buy_actions:
            st.markdown("### 🛒 Stocks to Buy/Increase")
            for action in buy_actions:
                st.markdown(f"""
                **{action['ticker'].split('.')[0]}**: Buy {action['change_shares']} additional shares
                - Current: {action['current_shares']} shares ({action['current_weight']:.2f}% of portfolio)
                - Target: {action['current_shares'] + action['change_shares']} shares ({action['target_weight']:.2f}% of portfolio)
                - Rationale: {action['rationale']}
                """)
        
        # Sell actions
        sell_actions = [rec for rec in recommendations['rebalance_suggestions'] if rec['action'] == 'SELL']
        if sell_actions:
            st.markdown("### 💸 Stocks to Sell/Reduce")
            for action in sell_actions:
                st.markdown(f"""
                **{action['ticker'].split('.')[0]}**: Sell {abs(action['change_shares'])} shares
                - Current: {action['current_shares']} shares ({action['current_weight']:.2f}% of portfolio)
                - Target: {action['current_shares'] + action['change_shares']} shares ({action['target_weight']:.2f}% of portfolio)
                - Rationale: {action['rationale']}
                """)
        
        # Hold actions
        hold_actions = [rec for rec in recommendations['rebalance_suggestions'] if rec['action'] == 'HOLD']
        if hold_actions:
            st.markdown("### ✋ Stocks to Hold")
            for action in hold_actions:
                st.markdown(f"""
                **{action['ticker'].split('.')[0]}**: Maintain current position of {action['current_shares']} shares ({action['current_weight']:.2f}% of portfolio)
                """)

def portfolio_analyzer_tab():
    """
    Complete implementation of the Portfolio Analyzer tab (Tab 4)
    with manual entry, CSV upload, and template download options
    """
    
    st.header("💼 Portfolio Analyzer")
    st.markdown("Upload your portfolio or enter stock details to analyze and optimize your holdings.")
    
    # Option to either upload a file or enter portfolio manually
    portfolio_input = st.radio("How would you like to enter your portfolio?", 
                             ["Enter manually", "Upload CSV"])
    
    if portfolio_input == "Enter manually":
        # Create a form for manual portfolio entry
        with st.form("portfolio_form"):
            st.subheader("Enter Your Portfolio")
            
            # Portfolio currency
            currency = st.radio("Portfolio Currency", ["INR (₹)", "USD ($)"], index=0)
            currency_symbol = "₹" if currency == "INR (₹)" else "$"
            
            # Create empty portfolio DataFrame
            if 'portfolio_df' not in st.session_state:
                st.session_state['portfolio_df'] = pd.DataFrame({
                    'Ticker': [''],
                    'Shares': [0],
                    'Avg Price': [0.0],
                    'Investment': [0.0]
                })
            
            # Display editable portfolio
            portfolio_data = st.data_editor(
                st.session_state['portfolio_df'],
                column_config={
                    "Ticker": st.column_config.TextColumn("Stock Ticker"),
                    "Shares": st.column_config.NumberColumn("Number of Shares", min_value=0, step=1),
                    "Avg Price": st.column_config.NumberColumn(f"Average Price ({currency_symbol})", min_value=0.0, format=f"{currency_symbol}%.2f")
                },
                num_rows="dynamic",
                key="portfolio_editor"
            )
            
            st.session_state['portfolio_df'] = portfolio_data
            # Analysis parameters
            st.subheader("Analysis Parameters")
            
            col1, col2 = st.columns(2)
                        
            with col1:
                portfolio_start_date = st.date_input("Select start date", 
                                                 value=datetime.now() - timedelta(days=365),
                                                 format="YYYY-MM-DD",
                                                 key='portfolio_start_date')
                portfolio_test_period = st.slider("Test period (months)", 1, 6, 1, key='portfolio_test_period')
            
            with col2:
                # Model selection options (simplified for portfolio analysis)
                model_options = [
                    "2. Traditional ML models only (faster)"
                ]
                portfolio_model_choice = st.selectbox("Model selection", model_options, index=0, key='portfolio_model')
                portfolio_model_choice = portfolio_model_choice.split(".")[0]
                
                # Indicator options (simplified for portfolio analysis)
                indicator_options = [
                    "2. Basic indicators only (faster)"
                ]
                portfolio_indicator_choice = st.selectbox("Technical indicators", indicator_options, index=0, key='portfolio_indicator')
                portfolio_indicator_choice = portfolio_indicator_choice.split(".")[0]
            
            submitted = st.form_submit_button("Analyze Portfolio")
            
            if submitted:
                # Calculate investment for each row
                for i, row in portfolio_data.iterrows():
                    if not pd.isna(row['Shares']) and not pd.isna(row['Avg Price']):
                        portfolio_data.at[i, 'Investment'] = row['Shares'] * row['Avg Price']
                        
                        # Add .NS suffix for Indian stocks if not already present
                        if currency == "INR (₹)" and not row['Ticker'].endswith('.NS'):
                            portfolio_data.at[i, 'Ticker'] = row['Ticker'] + ".NS"
                
                # Validate portfolio data
                total_investment = portfolio_data['Investment'].sum()
                st.metric(f"Total Portfolio Investment ({currency_symbol})", f"{currency_symbol}{total_investment:,.2f}")
            
                valid_portfolio = portfolio_data.dropna(subset=['Ticker'])
                valid_portfolio = valid_portfolio[valid_portfolio['Ticker'] != '']
                
                if len(valid_portfolio) == 0:
                    st.error("Please enter at least one valid stock in your portfolio.")
                else:
                    st.session_state['valid_portfolio'] = valid_portfolio
                    st.session_state['portfolio_currency'] = currency
                    st.session_state['portfolio_params'] = {
                        'start_date': portfolio_start_date.strftime('%Y-%m-%d'),
                        'end_date': datetime.now().strftime('%Y-%m-%d'),
                        'test_period': portfolio_test_period,
                        'model_choice': portfolio_model_choice,
                        'indicator_choice': portfolio_indicator_choice
                    }
    
    else:  # "Download and use template"
        # Portfolio currency
        currency = st.radio("Portfolio Currency", ["INR (₹)", "USD ($)"], index=0)
        currency_symbol = "₹" if currency == "INR (₹)" else "$"
        
        # Add CSV download functionality
        st.subheader("📋 Download Sample Portfolio Template")
        
        # Generate sample CSV template
        sample_data = {
            'Ticker': ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS'],
            'Shares': [10, 15, 5, 20, 25, 8, 30, 40],
            'Avg Price': [2450.75, 1675.30, 3725.60, 1450.25, 1020.50, 2540.75, 675.40, 875.25],
            'Investment': [24507.50, 25129.50, 18628.00, 29005.00, 25512.50, 20326.00, 20262.00, 35010.00]
        }
        
        # Create DataFrame and convert to CSV
        csv_df = pd.DataFrame(sample_data)
        csv = csv_df.to_csv(index=False)
        
        # Generate download link
        b64 = base64.b64encode(csv.encode()).decode()
        download_filename = "sample_portfolio.csv"
        href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}" class="btn">Download Sample Portfolio CSV</a>'
        
        # Display instructions
        st.markdown("""
        1. Download the sample CSV template below
        2. Edit the file with your portfolio details using any spreadsheet application
        3. Upload the modified file using the form below
        """)
        
        # Display download link with styling
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin:10px 0;">
            {href}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Add CSV upload functionality
        st.subheader("📤 Upload Your Modified Portfolio CSV")
        
        # Create file uploader
        uploaded_file = st.file_uploader("Upload your portfolio CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                portfolio_data = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['Ticker', 'Shares', 'Avg Price']
                if not all(col in portfolio_data.columns for col in required_cols):
                    st.error("The CSV file must contain columns: Ticker, Shares, Avg Price")
                else:
                    # Calculate investment for each row if missing
                    if 'Investment' not in portfolio_data.columns:
                        portfolio_data['Investment'] = portfolio_data['Shares'] * portfolio_data['Avg Price']
                    
                    # Add .NS suffix for Indian stocks if not already present
                    if currency == "INR (₹)":
                        for i, row in portfolio_data.iterrows():
                            if not row['Ticker'].endswith('.NS'):
                                portfolio_data.at[i, 'Ticker'] = row['Ticker'] + ".NS"
                    
                    # Display the loaded portfolio with proper formatting
                    st.subheader("📊 Loaded Portfolio")
                    
                    # Format the display with currency
                    display_df = portfolio_data.copy()
                    display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                    display_df['Investment'] = display_df['Investment'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                    
                    st.dataframe(display_df)
                    
                    # Display total investment
                    total_investment = portfolio_data['Investment'].sum()
                    st.metric(f"Total Portfolio Investment ({currency_symbol})", f"{currency_symbol}{total_investment:,.2f}")
                    
                    # Analysis parameters
                    st.subheader("Analysis Parameters")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        portfolio_start_date = st.date_input("Select start date", 
                                                          value=datetime.now() - timedelta(days=365),
                                                          format="YYYY-MM-DD",
                                                          key='portfolio_template_start_date')
                        portfolio_test_period = st.slider("Test period (months)", 1, 6, 1, key='portfolio_template_test_period')
                    
                    with col2:
                        # Model selection options (simplified for portfolio analysis)
                        model_options = [
                            "2. Traditional ML models only (faster)"
                        ]
                        portfolio_model_choice = st.selectbox("Model selection", model_options, index=0, key='portfolio_template_model')
                        portfolio_model_choice = portfolio_model_choice.split(".")[0]
                        
                        # Indicator options (simplified for portfolio analysis)
                        indicator_options = [
                            "2. Basic indicators only (faster)"
                        ]
                        portfolio_indicator_choice = st.selectbox("Technical indicators", indicator_options, index=0, key='portfolio_template_indicator')
                        portfolio_indicator_choice = portfolio_indicator_choice.split(".")[0]
                    
                    if st.button("Analyze Portfolio", key="analyze_template_portfolio"):
                        st.session_state['valid_portfolio'] = portfolio_data
                        st.session_state['portfolio_currency'] = currency
                        st.session_state['portfolio_params'] = {
                            'start_date': portfolio_start_date.strftime('%Y-%m-%d'),
                            'end_date': datetime.now().strftime('%Y-%m-%d'),
                            'test_period': portfolio_test_period,
                            'model_choice': portfolio_model_choice,
                            'indicator_choice': portfolio_indicator_choice
                        }
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
    
    # Process the portfolio analysis if valid data is available
    if 'valid_portfolio' in st.session_state:
        # Create a progress bar
        progress_bar = st.progress(0, text="Starting portfolio analysis...")
        
        portfolio_df = st.session_state['valid_portfolio']
        currency = st.session_state['portfolio_currency']
        params = st.session_state['portfolio_params']
        
        # Run the enhanced portfolio analysis
        try:
            # First, analyze portfolio holdings for recommendations
            results_df, portfolio_metrics, recommendations = analyze_portfolio_holdings(
                portfolio_df=portfolio_df,
                start_date=params['start_date'],
                end_date=params['end_date'],
                test_period=params['test_period'],
                model_choice=params['model_choice'],
                indicator_choice=params['indicator_choice']
            )
            
            # Clear progress bar
            progress_bar.empty()
            
            # Get currency symbol
            currency_symbol = "₹" if currency == "INR (₹)" else "$"
            
            # Display portfolio analysis and recommendations
            display_portfolio_analysis_and_recommendations(
                results_df=results_df,
                portfolio_metrics=portfolio_metrics,
                recommendations=recommendations,
                currency_symbol=currency_symbol
            )
            
            # Also run the standard rebalancing analysis for comparison
            rebalance_results = analyze_portfolio(
                portfolio_df=portfolio_df,
                start_date=params['start_date'],
                end_date=params['end_date'],
                test_period=params['test_period'],
                model_choice=params['model_choice'],
                indicator_choice=params['indicator_choice']
            )
            
            # Display comparison with standard rebalancing if desired
            if st.checkbox("Show alternative portfolio rebalancing analysis"):
                st.subheader("📊 Alternative Portfolio Rebalancing")
                
                original_results_df, total_current_value, total_pnl, total_pnl_percent, rebalance_df = rebalance_results
                
                # Format the DataFrame for display
                display_df = original_results_df.copy()
                
                # Format monetary values and percentages
                for col in ['Avg Price', 'Current Price', 'Investment', 'Current Value', 'P&L']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                
                if 'P&L %' in display_df.columns:
                    display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Display rebalancing recommendations
                st.subheader("💰 Alternative Portfolio Rebalancing Recommendations")
                
                if not rebalance_df.empty:
                    # Format for display
                    display_rebalance = rebalance_df.copy()
                    
                    # Format monetary values
                    for col in ['Current Value', 'Target Value', 'Change Value']:
                        if col in display_rebalance.columns:
                            display_rebalance[col] = display_rebalance[col].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                    
                    st.dataframe(display_rebalance, use_container_width=True)
            
            # Add option to download full analysis report
            st.subheader("📋 Download Analysis Report")
            
            # Create a summary report in JSON format
            report_data = {
                "Portfolio Summary": {
                    "Total Investment": f"{currency_symbol}{portfolio_metrics['total_investment']:,.2f}",
                    "Current Value": f"{currency_symbol}{portfolio_metrics['total_current_value']:,.2f}",
                    "Total P&L": f"{currency_symbol}{portfolio_metrics['total_pnl']:,.2f} ({portfolio_metrics['total_pnl_percent']:.2f}%)",
                    "Expected Return Next Month": f"{portfolio_metrics['total_expected_return_percent']:.2f}%",
                    "Stocks": len(portfolio_df)
                },
                "Recommendations": {
                    "Overall Strategy": recommendations['overall_strategy'],
                    "Expected Performance": recommendations['expected_performance'],
                    "Buy Recommendations": len(recommendations['buy_recommendations']),
                    "Sell Recommendations": len(recommendations['sell_recommendations']),
                    "Hold Recommendations": len(recommendations['hold_recommendations'])
                },
                "Stock Analysis": {}
            }
            
            # Add individual stock analysis to report
            for ticker, row in results_df.iterrows():
                stock_name = ticker.split('.')[0]
                report_data["Stock Analysis"][stock_name] = {
                    "Current Price": f"{currency_symbol}{row.get('Current Price', 0):,.2f}",
                    "Signal": row.get('Signal', 'UNKNOWN'),
                    "Expected Monthly Return": f"{row.get('Expected Monthly Return', 0):.2f}%",
                    "P&L": f"{currency_symbol}{row.get('P&L', 0):,.2f} ({row.get('P&L %', 0):.2f}%)"
                }
            
            # Convert to JSON
            report_json = pd.Series(report_data).to_json(indent=2)
            
            # Create a download button
            st.download_button(
                label="Download Portfolio Analysis Report",
                data=report_json,
                file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Error during portfolio analysis: {str(e)}")
            progress_bar.empty()






# def portfolio_analyzer_tab():
#     """
#     Complete implementation of the Portfolio Analyzer tab (Tab 4)
#     with manual entry, CSV upload, and template download options
#     """
#     st.header("💼 Portfolio Analyzer")
#     st.markdown("Upload your portfolio or enter stock details to analyze and optimize your holdings.")
    
#     # Option to either upload a file or enter portfolio manually
#     portfolio_input = st.radio("How would you like to enter your portfolio?", 
#                              ["Enter manually", "Upload CSV"])
    
#     if portfolio_input == "Enter manually":
#         # Create a form for manual portfolio entry
#         with st.form("portfolio_form"):
#             st.subheader("Enter Your Portfolio")
            
#             # Portfolio currency
#             currency = st.radio("Portfolio Currency", ["INR (₹)", "USD ($)"], index=0)
#             currency_symbol = "₹" if currency == "INR (₹)" else "$"
            
            
            
#             # Create empty portfolio DataFrame
#             if 'portfolio_df' not in st.session_state:
#                 st.session_state['portfolio_df'] = pd.DataFrame({
#                     'Ticker': [''],
#                     'Shares': [0],
#                     'Avg Price': [0.0],
#                     'Investment': [0.0]
#                 })
            
#             # Display editable portfolio
#             portfolio_data = st.data_editor(
#                 st.session_state['portfolio_df'],
#                 column_config={
#                     "Ticker": st.column_config.TextColumn("Stock Ticker"),
#                     "Shares": st.column_config.NumberColumn("Number of Shares", min_value=0, step=1),
#                     "Avg Price": st.column_config.NumberColumn(f"Average Price ({currency_symbol})", min_value=0.0, format=f"{currency_symbol}%.2f")
#                     # , "Investment": st.column_config.NumberColumn(f"Total Investment ({currency_symbol})", min_value=0.0, format=f"{currency_symbol}%.2f")
#                 },
#                 num_rows="dynamic",
#                 key="portfolio_editor"
#             )
            
#             st.session_state['portfolio_df'] = portfolio_data
#             # Analysis parameters
#             st.subheader("Analysis Parameters")
            
#             col1, col2 = st.columns(2)
                        
#             with col1:
#                 portfolio_start_date = st.date_input("Select start date", 
#                                                  value=datetime.now() - timedelta(days=365),
#                                                  format="YYYY-MM-DD",
#                                                  key='portfolio_start_date')
#                 portfolio_test_period = st.slider("Test period (months)", 1, 6, 1, key='portfolio_test_period')
            
#             with col2:
#                 # Model selection options (simplified for portfolio analysis)
#                 model_options = [
#                     "2. Traditional ML models only (faster)"
#                 ]
#                 portfolio_model_choice = st.selectbox("Model selection", model_options, index=0, key='portfolio_model')
#                 portfolio_model_choice = portfolio_model_choice.split(".")[0]
                
#                 # Indicator options (simplified for portfolio analysis)
#                 indicator_options = [
#                     "2. Basic indicators only (faster)"
#                 ]
#                 portfolio_indicator_choice = st.selectbox("Technical indicators", indicator_options, index=0, key='portfolio_indicator')
#                 portfolio_indicator_choice = portfolio_indicator_choice.split(".")[0]
            
#             submitted = st.form_submit_button("Analyze Portfolio")
            
#             if submitted:
            
#                 # Calculate investment for each row
#                 for i, row in portfolio_data.iterrows():
#                     if not pd.isna(row['Shares']) and not pd.isna(row['Avg Price']):
#                         portfolio_data.at[i, 'Investment'] = row['Shares'] * row['Avg Price']
#                         portfolio_data.at[i, 'Ticker'] = row['Ticker']+ ".NS"
                        
#                         print("code testing")
#                         print(portfolio_data.at[i, 'Ticker'] )
#                 # Validate portfolio data
#                 total_investment = portfolio_data['Investment'].sum()
#                 st.metric(f"Total Portfolio Investment ({currency_symbol})", f"{currency_symbol}{total_investment:,.2f}")
            
#                 valid_portfolio = portfolio_data.dropna(subset=['Ticker'])
#                 valid_portfolio = valid_portfolio[valid_portfolio['Ticker'] != '']
                
#                 if len(valid_portfolio) == 0:
#                     st.error("Please enter at least one valid stock in your portfolio.")
#                 else:
#                     st.session_state['valid_portfolio'] = valid_portfolio
#                     st.session_state['portfolio_currency'] = currency
#                     st.session_state['portfolio_params'] = {
#                         'start_date': portfolio_start_date.strftime('%Y-%m-%d'),
#                         'end_date': datetime.now().strftime('%Y-%m-%d'),
#                         'test_period': portfolio_test_period,
#                         'model_choice': portfolio_model_choice,
#                         'indicator_choice': portfolio_indicator_choice
#                     }
    

    
#     else:  # "Download and use template"
#         # Portfolio currency
#         currency = st.radio("Portfolio Currency", ["INR (₹)", "USD ($)"], index=0)
#         currency_symbol = "₹" if currency == "INR (₹)" else "$"
        
#         # Add CSV download functionality
#         st.subheader("📋 Download Sample Portfolio Template")
        
#         # Generate sample CSV template
#         sample_data = {
#             'Ticker': ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS'],
#             'Shares': [10, 15, 5, 20, 25, 8, 30, 40],
#             'Avg Price': [2450.75, 1675.30, 3725.60, 1450.25, 1020.50, 2540.75, 675.40, 875.25],
#             'Investment': [24507.50, 25129.50, 18628.00, 29005.00, 25512.50, 20326.00, 20262.00, 35010.00]
#         }
        
#         # Create DataFrame and convert to CSV
#         csv_df = pd.DataFrame(sample_data)
#         csv = csv_df.to_csv(index=False)
        
#         # Generate download link
#         b64 = base64.b64encode(csv.encode()).decode()
#         download_filename = "sample_portfolio.csv"
#         href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}" class="btn">Download Sample Portfolio CSV</a>'
        
#         # Display instructions
#         st.markdown("""
#         1. Download the sample CSV template below
#         2. Edit the file with your portfolio details using any spreadsheet application
#         3. Upload the modified file using the form below
#         """)
        
#         # Display download link with styling
#         st.markdown(f"""
#         <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin:10px 0;">
#             {href}
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown("---")
        
#         # Add CSV upload functionality
#         st.subheader("📤 Upload Your Modified Portfolio CSV")
        
#         # Create file uploader
#         uploaded_file = st.file_uploader("Upload your portfolio CSV file", type=['csv'])
        
#         if uploaded_file is not None:
#             try:
#                 # Read CSV file
#                 portfolio_data = pd.read_csv(uploaded_file)
                
#                 # Check required columns
#                 required_cols = ['Ticker', 'Shares', 'Avg Price']
#                 if not all(col in portfolio_data.columns for col in required_cols):
#                     st.error("The CSV file must contain columns: Ticker, Shares, Avg Price")
#                 else:
#                     # Calculate investment for each row if missing
#                     if 'Investment' not in portfolio_data.columns:
#                         portfolio_data['Investment'] = portfolio_data['Shares'] * portfolio_data['Avg Price']
                    
#                     # Display the loaded portfolio with proper formatting
#                     st.subheader("📊 Loaded Portfolio")
                    
#                     # Format the display with currency
#                     display_df = portfolio_data.copy()
#                     display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
#                     display_df['Investment'] = display_df['Investment'].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                    
#                     st.dataframe(display_df)
                    
#                     # Display total investment
#                     total_investment = portfolio_data['Investment'].sum()
#                     st.metric(f"Total Portfolio Investment ({currency_symbol})", f"{currency_symbol}{total_investment:,.2f}")
                    
#                     # Analysis parameters
#                     st.subheader("Analysis Parameters")
                    
#                     col1, col2 = st.columns(2)
                    
#                     with col1:
#                         portfolio_start_date = st.date_input("Select start date", 
#                                                           value=datetime.now() - timedelta(days=365),
#                                                           format="YYYY-MM-DD",
#                                                           key='portfolio_template_start_date')
#                         portfolio_test_period = st.slider("Test period (months)", 1, 6, 1, key='portfolio_template_test_period')
                    
#                     with col2:
#                         # Model selection options (simplified for portfolio analysis)
#                         model_options = [
#                             "2. Traditional ML models only (faster)"
#                         ]
#                         portfolio_model_choice = st.selectbox("Model selection", model_options, index=0, key='portfolio_template_model')
#                         portfolio_model_choice = portfolio_model_choice.split(".")[0]
                        
#                         # Indicator options (simplified for portfolio analysis)
#                         indicator_options = [
#                             "2. Basic indicators only (faster)"
#                         ]
#                         portfolio_indicator_choice = st.selectbox("Technical indicators", indicator_options, index=0, key='portfolio_template_indicator')
#                         portfolio_indicator_choice = portfolio_indicator_choice.split(".")[0]
                    
#                     if st.button("Analyze Portfolio", key="analyze_template_portfolio"):
#                         st.session_state['valid_portfolio'] = portfolio_data
#                         st.session_state['portfolio_currency'] = currency
#                         st.session_state['portfolio_params'] = {
#                             'start_date': portfolio_start_date.strftime('%Y-%m-%d'),
#                             'end_date': datetime.now().strftime('%Y-%m-%d'),
#                             'test_period': portfolio_test_period,
#                             'model_choice': portfolio_model_choice,
#                             'indicator_choice': portfolio_indicator_choice
#                         }
#             except Exception as e:
#                 st.error(f"Error loading CSV file: {str(e)}")
    
#     # Process the portfolio analysis if valid data is available
#     if 'valid_portfolio' in st.session_state:
#         # Create a progress bar
#         progress_bar = st.progress(0, text="Starting portfolio analysis...")
        
#         portfolio_df = st.session_state['valid_portfolio']
#         currency = st.session_state['portfolio_currency']
#         params = st.session_state['portfolio_params']
        
#         # Run the portfolio analysis
#         try:
#             results_df, total_current_value, total_pnl, total_pnl_percent, rebalance_df = analyze_portfolio(
#                 portfolio_df=portfolio_df,
#                 start_date=params['start_date'],
#                 end_date=params['end_date'],
#                 test_period=params['test_period'],
#                 model_choice=params['model_choice'],
#                 indicator_choice=params['indicator_choice']
#             )
            
#             # Clear progress bar
#             progress_bar.empty()
            
#             # Display portfolio summary
#             st.subheader("📊 Portfolio Summary")
            
#             # Currency symbol for display
#             currency_symbol = "₹" if currency == "INR (₹)" else "$"
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Total Investment", f"{currency_symbol}{portfolio_df['Investment'].sum():,.2f}")
#             with col2:
#                 st.metric("Current Value", f"{currency_symbol}{total_current_value:,.2f}")
#             with col3:
#                 st.metric("Total P&L", f"{currency_symbol}{total_pnl:,.2f} ({total_pnl_percent:.2f}%)", 
#                           delta=f"{total_pnl_percent:.2f}%")
            
#             # Display detailed results
#             st.subheader("📊 Portfolio Performance")
            
#             if not results_df.empty and 'Current Value' in results_df.columns:
#                 # Format the DataFrame for display
#                 display_df = results_df.copy()
                
#                 # Format monetary values and percentages
#                 for col in ['Avg Price', 'Current Price', 'Investment', 'Current Value', 'P&L']:
#                     if col in display_df.columns:
#                         display_df[col] = display_df[col].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                
#                 if 'P&L %' in display_df.columns:
#                     display_df['P&L %'] = display_df['P&L %'].apply(lambda x: f"{x:.2f}%")
                
#                 st.dataframe(display_df, use_container_width=True)
                
#                 # Plot portfolio composition
#                 st.subheader("📊 Portfolio Composition")
                
#                 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
#                 # Plot investment allocation
#                 investment_data = portfolio_df[['Ticker', 'Investment']].copy()
#                 investment_data['Percentage'] = investment_data['Investment'] / investment_data['Investment'].sum() * 100
                
#                 ax1.pie(investment_data['Percentage'], labels=investment_data['Ticker'], autopct='%1.1f%%')
#                 ax1.set_title('Investment Allocation')
                
#                 # Plot current value allocation (if available)
#                 if 'Current Value' in results_df.columns:
#                     value_data = results_df[['Current Value']].copy()
#                     value_data['Percentage'] = value_data['Current Value'] / value_data['Current Value'].sum() * 100
                    
#                     ax2.pie(value_data['Percentage'], labels=value_data.index, autopct='%1.1f%%')
#                     ax2.set_title('Current Value Allocation')
                
#                 plt.tight_layout()
#                 st.pyplot(fig)
                
#                 # Plot P&L by stock
#                 st.subheader("📊 Profit & Loss by Stock")
                
#                 if 'P&L' in results_df.columns:
#                     fig, ax = plt.subplots(figsize=(10, 6))
                    
#                     # Create a bar chart
#                     bars = ax.bar(results_df.index, results_df['P&L'])
                    
#                     # Color bars based on positive/negative P&L
#                     for i, bar in enumerate(bars):
#                         if results_df['P&L'].iloc[i] < 0:
#                             bar.set_color('red')
#                         else:
#                             bar.set_color('green')
                    
#                     ax.set_title('Profit & Loss by Stock')
#                     ax.set_xlabel('Stock')
#                     ax.set_ylabel(f'P&L ({currency_symbol})')
#                     ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
#                     plt.xticks(rotation=45)
#                     plt.tight_layout()
#                     st.pyplot(fig)
                
#                 # Show trading signals
#                 st.subheader("🚨 Trading Signals")
                
#                 if 'Signal' in results_df.columns:
#                     # Create a dataframe for displaying signals
#                     signals_df = results_df[['Signal', 'Current Price', 'Best Model']].copy()
                    
#                     # Count signals
#                     buy_count = (signals_df['Signal'] == 'BUY').sum()
#                     sell_count = (signals_df['Signal'] == 'SELL').sum()
#                     hold_count = (signals_df['Signal'] == 'HOLD').sum()
#                     unknown_count = len(signals_df) - buy_count - sell_count - hold_count
                    
#                     # Display signal counts
#                     col1, col2, col3, col4 = st.columns(4)
#                     with col1:
#                         st.metric("Buy Signals", buy_count)
#                     with col2:
#                         st.metric("Sell Signals", sell_count)
#                     with col3:
#                         st.metric("Hold Signals", hold_count)
#                     with col4:
#                         st.metric("Unknown Signals", unknown_count)
                    
#                     # Display signals table
#                     st.dataframe(signals_df, use_container_width=True)
                    
#                     # Display rebalancing recommendations
#                     st.subheader("💰 Portfolio Rebalancing Recommendations")
                    
#                     if not rebalance_df.empty:
#                         # Format for display
#                         display_rebalance = rebalance_df.copy()
                        
#                         # Format monetary values
#                         for col in ['Current Value', 'Target Value', 'Change Value']:
#                             if col in display_rebalance.columns:
#                                 display_rebalance[col] = display_rebalance[col].apply(lambda x: f"{currency_symbol}{x:,.2f}")
                        
#                         st.dataframe(display_rebalance, use_container_width=True)
                        
#                         # Show rebalancing actions
#                         st.subheader("📝 Action Plan")
                        
#                         # Buy actions
#                         buy_actions = rebalance_df[rebalance_df['Action'] == 'BUY']
#                         if not buy_actions.empty:
#                             st.markdown("### 🛒 Stocks to Buy")
#                             for ticker, row in buy_actions.iterrows():
#                                 st.markdown(f"""
#                                 **{ticker}**: Buy {int(row['Change Shares'])} shares at approximately {currency_symbol}{results_df.loc[ticker, 'Current Price']:.2f} per share
#                                 - Current Position: {int(row['Current Shares'])} shares
#                                 - Target Position: {int(row['New Shares'])} shares
#                                 - Approx. Investment: {currency_symbol}{row['Change Shares'] * results_df.loc[ticker, 'Current Price']:,.2f}
#                                 """)
                        
#                         # Sell actions
#                         sell_actions = rebalance_df[rebalance_df['Action'] == 'SELL']
#                         if not sell_actions.empty:
#                             st.markdown("### 💸 Stocks to Sell")
#                             for ticker, row in sell_actions.iterrows():
#                                 st.markdown(f"""
#                                 **{ticker}**: Sell {abs(int(row['Change Shares']))} shares at approximately {currency_symbol}{results_df.loc[ticker, 'Current Price']:.2f} per share
#                                 - Current Position: {int(row['Current Shares'])} shares
#                                 - Target Position: {int(row['New Shares'])} shares
#                                 - Approx. Value: {currency_symbol}{abs(row['Change Shares']) * results_df.loc[ticker, 'Current Price']:,.2f}
#                                 """)
                        
#                         # Hold actions
#                         hold_actions = rebalance_df[rebalance_df['Action'] == 'HOLD']
#                         if not hold_actions.empty:
#                             st.markdown("### ✋ Stocks to Hold")
#                             for ticker, row in hold_actions.iterrows():
#                                 st.markdown(f"""
#                                 **{ticker}**: Hold current position of {int(row['Current Shares'])} shares
#                                 """)
                        
#                         # Estimated new portfolio performance
#                         st.subheader("📊 Estimated New Portfolio Performance")
                        
#                         # Calculate expected return after rebalancing
#                         total_new_investment = 0
#                         total_expected_return = 0
                        
#                         for ticker, row in rebalance_df.iterrows():
#                             if 'Signal' in results_df.columns and ticker in results_df.index:
#                                 new_shares = row['New Shares']
#                                 current_price = results_df.loc[ticker, 'Current Price']
#                                 new_investment = new_shares * current_price
                                
#                                 # Estimate expected return based on signal
#                                 expected_return_rate = 0.05 if results_df.loc[ticker, 'Signal'] == 'BUY' else 0.01  # Simplified assumption
#                                 expected_return = new_investment * expected_return_rate
                                
#                                 total_new_investment += new_investment
#                                 total_expected_return += expected_return
                        
#                         # Display expected performance
#                         col1, col2, col3 = st.columns(3)
#                         with col1:
#                             st.metric("New Portfolio Value", f"{currency_symbol}{total_new_investment:,.2f}")
#                         with col2:
#                             st.metric("Expected 30-Day Return", f"{currency_symbol}{total_expected_return:,.2f}")
#                         with col3:
#                             expected_return_percent = (total_expected_return / total_new_investment * 100) if total_new_investment > 0 else 0
#                             st.metric("Expected Return Rate", f"{expected_return_percent:.2f}%")
#                     else:
#                         st.info("No rebalancing recommendations available. Please check your portfolio data.")
                
#                 # Download portfolio analysis report
#                 st.subheader("📋 Download Analysis Report")
                
#                 # Create a summary report
#                 report_data = {
#                     "Portfolio Summary": {
#                         "Total Investment": f"{currency_symbol}{portfolio_df['Investment'].sum():,.2f}",
#                         "Current Value": f"{currency_symbol}{total_current_value:,.2f}",
#                         "Total P&L": f"{currency_symbol}{total_pnl:,.2f} ({total_pnl_percent:.2f}%)",
#                         "Stocks": len(portfolio_df)
#                     },
#                     "Trading Signals": {
#                         "Buy": buy_count,
#                         "Sell": sell_count,
#                         "Hold": hold_count,
#                         "Unknown": unknown_count
#                     },
#                     "Rebalancing Summary": {
#                         "Stocks to Buy": len(rebalance_df[rebalance_df['Action'] == 'BUY']),
#                         "Stocks to Sell": len(rebalance_df[rebalance_df['Action'] == 'SELL']),
#                         "Stocks to Hold": len(rebalance_df[rebalance_df['Action'] == 'HOLD'])
#                     }
#                 }
                
#                 # Convert to JSON
#                 report_json = pd.Series(report_data).to_json()
                
#                 # Create a download button
#                 st.download_button(
#                     label="Download Portfolio Analysis Report",
#                     data=report_json,
#                     file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.json",
#                     mime="application/json"
#                 )
#             else:
#                 st.warning("No detailed results available. Please check your portfolio data.")
#         except Exception as e:
#             st.error(f"Error during portfolio analysis: {str(e)}")
#             progress_bar.empty()

# # This function should be called in the main() function under the tab4 section:
# # with tab4:
# #     portfolio_analyzer_tab()




def create_sample_portfolio_csv():
    """
    Create a sample portfolio CSV template with example stocks
    """
    # Create sample data
    sample_data = {
        'Ticker': ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS'],
        'Shares': [10, 15, 5, 20, 25, 8, 30, 40],
        'Avg Price': [2450.75, 1675.30, 3725.60, 1450.25, 1020.50, 2540.75, 675.40, 875.25],
        'Investment': [24507.50, 25129.50, 18628.00, 29005.00, 25512.50, 20326.00, 20262.00, 35010.00]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Convert DataFrame to CSV
    csv = df.to_csv(index=False)
    
    return csv

def get_csv_download_link(csv_string, file_name="sample_portfolio.csv"):
    """
    Generate a download link for a CSV file
    """
    # Convert string to bytes
    b64 = base64.b64encode(csv_string.encode()).decode()
    
    # Generate download link
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    
    return href

def implement_portfolio_csv_download():
    """
    Add CSV download functionality to the Portfolio tab
    """
    st.subheader("📋 Download Sample Portfolio Template")
    
    # Generate sample CSV template
    csv_template = create_sample_portfolio_csv()
    
    # Create download link
    download_link = get_csv_download_link(csv_template)
    
    # Display instructions
    st.markdown("""
    1. Download the sample CSV template
    2. Edit the file with your portfolio details
    3. Upload the modified file using the form below
    """)
    
    # Display download link
    st.markdown(download_link, unsafe_allow_html=True)
    
    return csv_template

def implement_portfolio_csv_upload():
    """
    Add CSV upload functionality to the Portfolio tab
    """

    implement_portfolio_csv_download()


    st.subheader("📤 Upload Your Portfolio CSV")
    
    # Create file uploader
    uploaded_file = st.file_uploader("Upload your portfolio CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            portfolio_df = pd.read_csv(uploaded_file)
            
            # Check if the required columns exist
            required_cols = ['Ticker', 'Shares', 'Avg Price']
            if not all(col in portfolio_df.columns for col in required_cols):
                st.error("The CSV file must contain columns: Ticker, Shares, Avg Price")
                return None
            
            # Calculate investment if not present
            if 'Investment' not in portfolio_df.columns:
                portfolio_df['Investment'] = portfolio_df['Shares'] * portfolio_df['Avg Price']
            
            # Display the loaded portfolio
            st.success("Portfolio successfully loaded!")
            st.dataframe(portfolio_df)
            
            # Store in session state
            st.session_state['valid_portfolio'] = portfolio_df
            
            return portfolio_df
        
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None
    
    return None

# Function to add both download and upload features to the Portfolio tab
def add_portfolio_csv_features():
    """
    Add both download and upload features to the Portfolio tab
    """
    # Add a section for CSV template download
    csv_template = implement_portfolio_csv_download()
    
    # Add a separator
    st.markdown("---")
    
    # Add a section for CSV upload
    portfolio_df = implement_portfolio_csv_upload()
    
    return portfolio_df















# Define functions for the app

def run_nifty50_analysis(test_period, investment, start_date, max_workers):
    """Run analysis on Nifty 50 stocks"""
    with st.spinner('Analyzing Nifty 50 stocks... This may take several minutes.'):
        analyzer = Nifty50Analyzer(
            test_period_months=test_period,
            initial_investment=investment,
            start_date=start_date
        )
        recommendations = analyzer.run_analysis(max_workers=max_workers)
        return analyzer, recommendations

def analyze_single_stock(ticker, start_date, end_date, test_period, investment, model_choice, indicator_choice, target_choice):
    """Run detailed analysis on a single stock"""
    with st.spinner(f'Analyzing {ticker}... This may take a minute.'):
        try:
            # Create StockModelComparison instance
            stock_model = StockModelComparison(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                test_period_months=test_period,
                initial_investment=investment
            )
            
            # Configure based on user preferences
            stock_model.set_enabled_models(model_choice)
            stock_model.set_indicator_level(indicator_choice)
            stock_model.set_target_type(target_choice)
            
            # Run the analysis
            comparison_df, performance, test_performance, best_model = stock_model.run_advanced_analysis()    
            
            return stock_model, comparison_df, performance,test_performance,  best_model
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")
            return None, None, None, None, None
        





def analyze_portfolio(portfolio_df, start_date, end_date, test_period, model_choice, indicator_choice):
    """Analyze a portfolio of stocks"""
    results = {}
    
    total_investment = portfolio_df['Investment'].sum()
    rebalance_total = total_investment

    progress_bar = st.progress(0, text="Starting portfolio analysis...")
    
    with st.spinner('Analyzing your portfolio... This may take several minutes.'):
        for idx, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            shares = row['Shares']
            avg_price = row['Avg Price']
            investment = row['Investment']
            
            progress_bar.progress((idx + 1) / len(portfolio_df), text=f"Analyzing {ticker}...")
            
            try:
                # Create StockModelComparison instance
                stock_model = StockModelComparison(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    test_period_months=test_period,
                    initial_investment=investment
                )
                
                # Configure based on user preferences
                stock_model.set_enabled_models(model_choice)
                stock_model.set_indicator_level(indicator_choice)
                stock_model.set_target_type("1")  # Next day direction for portfolio analysis
                
                # Run the basic analysis
                stock_model.download_data()
                stock_model.calculate_technical_indicators()
                stock_model.prepare_features()
                stock_model.build_models()
                stock_model.train_evaluate_models()
                stock_model.generate_trading_signals()
                
                # Get last price and signal
                last_price = stock_model.stock_data['Close'].iloc[-1]
                latest_signal = stock_model.stock_data['ML_Signal'].iloc[-1] if 'ML_Signal' in stock_model.stock_data.columns else None
                
                # Calculate current value and P&L
                current_value = shares * last_price
                pnl = current_value - investment
                pnl_percent = (pnl / investment) * 100
                
                # Add to results
                results[ticker] = {
                    'Ticker': ticker,
                    'Shares': shares,
                    'Avg Price': avg_price,
                    'Investment': investment,
                    'Current Price': last_price,
                    'Current Value': current_value,
                    'P&L': pnl,
                    'P&L %': pnl_percent,
                    'Signal': 'BUY' if latest_signal == 1 else 'SELL' if latest_signal == 0 else 'HOLD' if latest_signal == 0.5 else 'UNKNOWN',
                    'Best Model': stock_model.best_model_name,
                    'Last Price': last_price,
                }
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                results[ticker] = {
                    'Ticker': ticker,
                    'Shares': shares,
                    'Avg Price': avg_price,
                    'Investment': investment,
                    'Error': str(e)
                }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Calculate portfolio metrics
    total_current_value = results_df['Current Value'].sum() if 'Current Value' in results_df.columns else 0
    total_pnl = results_df['P&L'].sum() if 'P&L' in results_df.columns else 0
    total_pnl_percent = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
    
    
    # Calculate rebalancing recommendations
    rebalance_recommendations = {}
    
    if not results_df.empty and 'Signal' in results_df.columns:
        # Calculate ideal weights
        total_stocks = len(results_df)
        buy_signals = results_df[results_df['Signal'] == 'BUY']
        sell_signals = results_df[results_df['Signal'] == 'SELL']
        hold_signals = results_df[results_df['Signal'] == 'HOLD']
#        unknown_signals = results_df[~results_df['Signal'].isin(['BUY', 'SELL', 'HOLD'])]        

        buy_count = len(buy_signals)
        sell_count = len(sell_signals)
        hold_count = len(hold_signals)
#        unknown_count = len(unknown_signals)
            
        if buy_count > 0:
            # Allocate more to buy signals
            buy_weight = 0.7 if buy_count < total_stocks else 1.0
            remaining_weight  = 1.0 - buy_weight
            
            hold_ratio = 0.8
            sell_ratio = 0.2
            
            hold_weight = remaining_weight * hold_ratio if hold_count > 0 else 0
            sell_weight = remaining_weight * sell_ratio if sell_count > 0 else 0            
            
            if hold_count == 0 and sell_count > 0:
                sell_weight = remaining_weight
            elif sell_count == 0 and hold_count > 0:
                hold_weight = remaining_weight            
            
            buy_per_stock = (rebalance_total * buy_weight) / buy_count if buy_count > 0 else 0
            hold_per_stock = (rebalance_total * hold_weight) / hold_count if hold_count > 0 else 0
            sell_per_stock = (rebalance_total * sell_weight) / sell_count if sell_count > 0 else 0
                    
            
            for ticker, row in results_df.iterrows():
                current_value = row['Current Value'] if 'Current Value' in row else 0
                current_shares = row['Shares']
                
                if row['Signal'] == 'BUY':
                    target_value = buy_per_stock
                    signal_strength = 'Strong Buy'
                
                elif row['Signal'] == 'HOLD':
                    target_value = min(current_value, hold_per_stock * 1.2)  # Allow slightly higher than target
                    target_value = max(target_value, hold_per_stock * 0.8)   # But not too low
                    signal_strength = 'Hold'
                    
                elif row['Signal'] == 'SELL':
                    target_value = sell_per_stock
                    # Consider model strength for sell decisions
                    model_name = row['Best Model'] if 'Best Model' in row else 'Unknown'
                    if model_name in ['XGBoost', 'Random Forest', 'LSTM', 'Ensemble']:
                        target_value = min(target_value, sell_per_stock * 0.5)  # Strong models -> more reduction
                    signal_strength = 'Strong Sell'
                    
                else:
                    # Unknown signals treated like HOLD but with less confidence
                    target_value = current_value * 0.9  # Slight reduction
                    signal_strength = 'Hold (Low Confidence)'
                
                # Calculate change in value and shares
                change_value = target_value - current_value

                    
                change_shares = int(change_value / row['Last Price']) if 'Last Price' in row and row['Last Price'] > 0 else 0

                # Calculate share change based on current price
                if 'Last Price' in row and row['Last Price'] > 0:
                    change_shares = int(change_value / row['Last Price'])
                    
                    # Special case for HOLD signals - minimize unnecessary trading
                    if row['Signal'] == 'HOLD':
                        # Only trade if change is significant (>10% of position)
                        if abs(change_shares) < current_shares * 0.1:
                            change_shares = 0  # Don't bother with small adjustments
                    
                    # Special case for SELL signals
                    if row['Signal'] == 'SELL' and target_value < current_value * 0.4:
                        change_shares = -current_shares  # Complete liquidation
                        target_value = 0
                else:
                    change_shares = 0
                
                # Calculate new shares
                new_shares = current_shares + change_shares
                
                # Determine action based on share change
                if change_shares > current_shares * 0.3:
                    action = 'BUY MORE'
                elif change_shares > 0:
                    action = 'BUY'
                elif change_shares < 0:
                    if change_shares == -current_shares:
                        action = 'SELL ALL'
                    elif abs(change_shares) > current_shares * 0.7:
                        action = 'SELL MOST'
                    elif abs(change_shares) > current_shares * 0.3:
                        action = 'SELL SOME'
                    else:
                        action = 'REDUCE SLIGHTLY'
                else:
                    action = 'HOLD'
                    
                # For HOLD signals with zero change_shares, always show HOLD action
                if row['Signal'] == 'HOLD' and change_shares == 0:
                    action = 'HOLD'
                    signal_strength = 'Maintain Position'
                
                rebalance_recommendations[ticker] = {
                    'Ticker': ticker,
                    'Current Value': current_value,
                    'Target Value': target_value,
                    'Change Value': change_value,
                    'Current Shares': current_shares,
                    'Change Shares': change_shares,
                    'New Shares': new_shares,
                    'Action': action,
                    'Recommendation': row['Signal'],
                    'Signal Strength': signal_strength
                }    
    rebalance_df = pd.DataFrame.from_dict(rebalance_recommendations, orient='index')
    
    return results_df, total_current_value, total_pnl, total_pnl_percent, rebalance_df

# Set up the main app structure
def main():
    st.title("📈 Advanced Stock Market Analyzer")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Nifty 50 Analysis", "Individual Stock Analysis", "Custom Stock Analysis", "Portfolio Analyzer"])
    
    with tab1:
        st.header("🇮🇳 Nifty 50 Market Analysis")
        st.markdown("Analyze all Nifty 50 stocks to find the best performers for the next 5-day period.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Select start date for historical data", 
                                      value=datetime.now() - timedelta(days=365*2),
                                      format="YYYY-MM-DD")
            test_period = st.slider("Test period (months)", 1, 12, 1)
        
        with col2:
            investment = st.number_input("Initial investment amount (₹)", 
                                        min_value=10000, 
                                        max_value=10000000, 
                                        value=100000,
                                        step=10000)
            max_workers = st.slider("Number of parallel processes", 1, 8, 4)
        
        if st.button("Run Nifty 50 Analysis", key='nifty_analysis'):
            analyzer, recommendations = run_nifty50_analysis(
                test_period=test_period,
                investment=investment,
                start_date=start_date.strftime('%Y-%m-%d'),
                max_workers=max_workers
            )
            
            if recommendations:
                st.session_state['nifty_recommendations'] = recommendations
                st.session_state['nifty_analyzer'] = analyzer
                st.success("Analysis completed!")
            else:
                st.error("Analysis failed. Please check the logs.")
        
        if 'nifty_recommendations' in st.session_state:
            recommendations = st.session_state['nifty_recommendations']
            analyzer = st.session_state['nifty_analyzer']
            
            st.subheader("🏆 Top Nifty 50 Stocks to Buy for Next 5 Days")
            
            top_buy = recommendations['top_buy']
            if not top_buy.empty:
                # Format the DataFrame for display
                display_df = top_buy[['last_price', 'test_return', 'alpha', 'sharpe', 'win_rate', 'recommendation']].copy()
                
                # Format percentages
                display_df['test_return'] = display_df['test_return'].apply(lambda x: f"{x*100:.2f}%")
                display_df['alpha'] = display_df['alpha'].apply(lambda x: f"{x*100:.2f}%")
                display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x*100:.2f}%")
                
                # Rename columns
                display_df.columns = ['Price (₹)', 'Test Return', 'Alpha', 'Sharpe', 'Win Rate', 'Signal']
                
                # Format index (remove .NS)
                display_df.index = [idx.split('.')[0] for idx in display_df.index]
                
                st.dataframe(display_df, use_container_width=True)
                
                # Show comparison plots
                if len(top_buy) >= 3:
                    st.subheader("📊 Performance Comparison")
                    
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Format index (remove .NS)
                    plot_data = top_buy.copy()
                    plot_data.index = [idx.split('.')[0] for idx in plot_data.index]
                    
                    # Plot test returns
                    sns.barplot(x=plot_data.index, y=plot_data['test_return'] * 100, ax=axes[0, 0])
                    axes[0, 0].set_title('Test Period Returns (%)')
                    axes[0, 0].set_ylabel('Return (%)')
                    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45)
                    
                    # Plot alpha
                    sns.barplot(x=plot_data.index, y=plot_data['alpha'] * 100, ax=axes[0, 1])
                    axes[0, 1].set_title('Alpha')
                    axes[0, 1].set_ylabel('Alpha (%)')
                    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45)
                    
                    # Plot Sharpe ratio
                    sns.barplot(x=plot_data.index, y=plot_data['sharpe'], ax=axes[1, 0])
                    axes[1, 0].set_title('Sharpe Ratio')
                    axes[1, 0].set_ylabel('Sharpe Ratio')
                    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)
                    
                    # Plot win rate
                    sns.barplot(x=plot_data.index, y=plot_data['win_rate'] * 100, ax=axes[1, 1])
                    axes[1, 1].set_title('Win Rate')
                    axes[1, 1].set_ylabel('Win Rate (%)')
                    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Plot expected 5-day returns
                    st.subheader("📈 Expected 5-Day Returns")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    returns_data = plot_data['potential_5day_return'] * 100
                    sns.barplot(x=plot_data.index, y=returns_data, ax=ax)
                    ax.set_title('Expected 5-Day Returns for Top Picks (%)')
                    ax.set_ylabel('Expected Return (%)')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Show investment allocation
                st.subheader("💰 Investment Allocation")
                
                # Calculate allocation
                investment_per_stock = investment / len(top_buy)
                
                allocation_data = []
                total_actual_investment = 0
                total_expected_return = 0
                
                for ticker, row in top_buy.iterrows():
                    company_name = ticker.split('.')[0]
                    shares = int(investment_per_stock / row['last_price'])
                    actual_investment = shares * row['last_price']
                    expected_gain = actual_investment * row['potential_5day_return']
                    
                    allocation_data.append({
                        'Company': company_name,
                        'Shares': shares,
                        'Price (₹)': f"₹{row['last_price']:.2f}",
                        'Investment (₹)': f"₹{actual_investment:,.2f}",
                        'Expected Gain (₹)': f"₹{expected_gain:,.2f}",
                        'Expected Return (%)': f"{row['potential_5day_return']*100:.2f}%"
                    })
                    
                    total_actual_investment += actual_investment
                    total_expected_return += expected_gain
                
                st.table(pd.DataFrame(allocation_data))
                
                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Investment", f"₹{total_actual_investment:,.2f}")
                with col2:
                    st.metric("Expected 5-Day Gain", f"₹{total_expected_return:,.2f}")
                with col3:
                    st.metric("Expected Return", f"{total_expected_return/total_actual_investment*100:.2f}%")
            else:
                st.warning("No buy recommendations available based on the analysis.")
            
            # Show overall performance ranking
            st.subheader("📈 Overall Nifty 50 Performance Ranking (Top 10)")
            
            top_overall = recommendations['top_overall']
            if not top_overall.empty:
                # Format the DataFrame for display
                cols_to_display = ['last_price', 'test_return', 'alpha', 'sharpe', 'win_rate', 'recommendation']
                available_cols = [col for col in cols_to_display if col in top_overall.columns]
                
                if available_cols:
                    display_df = top_overall[available_cols].copy()
                    
                    # Format percentages
                    if 'test_return' in display_df.columns:
                        display_df['test_return'] = display_df['test_return'].apply(lambda x: f"{x*100:.2f}%")
                    if 'alpha' in display_df.columns:
                        display_df['alpha'] = display_df['alpha'].apply(lambda x: f"{x*100:.2f}%")
                    if 'win_rate' in display_df.columns:
                        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x*100:.2f}%")
                    
                    # Rename columns
                    col_mapping = {
                        'last_price': 'Price (₹)', 
                        'test_return': 'Test Return', 
                        'alpha': 'Alpha', 
                        'sharpe': 'Sharpe', 
                        'win_rate': 'Win Rate', 
                        'recommendation': 'Signal'
                    }
                    display_df.columns = [col_mapping.get(col, col) for col in display_df.columns]
                    
                    # Format index (remove .NS)
                    display_df.index = [idx.split('.')[0] for idx in display_df.index]
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.warning("No data available to display.")
            else:
                st.warning("No overall performance data available.")
    
    with tab2:
        st.header("🔍 Nifty 50 Stock Details")
        st.markdown("Select a stock from the dropdown to see detailed analysis.")
        
        # Get the list of Nifty 50 stocks
        nifty50_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 
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
        
        # Create a dropdown with company names without .NS
        stock_options = [stock.split('.')[0] for stock in nifty50_stocks]
        selected_stock_name = st.selectbox("Select a stock", stock_options)
        
        # Add .NS to get the actual ticker
        selected_stock = f"{selected_stock_name}.NS"
        
        col1, col2 = st.columns(2)
        
        with col1:
            detail_start_date = st.date_input("Select start date", 
                                           value=datetime.now() - timedelta(days=365*2),
                                           format="YYYY-MM-DD",
                                           key='detail_start_date')
            detail_test_period = st.slider("Test period (months)", 1, 12, 6, key='detail_test_period')
        
        with col2:
            detail_investment = st.number_input("Initial investment amount (₹)", 
                                             min_value=10000, 
                                             max_value=10000000, 
                                             value=100000,
                                             step=10000,
                                             key='detail_investment')
            
            # Add model selection options
            model_options = [
                "1. All models (default)",
                "2. Traditional ML models only (faster)",
                "3. Deep learning models only"
            ]
            detail_model_choice = st.selectbox("Model selection", model_options, index=1)
            detail_model_choice = detail_model_choice.split(".")[0]
            
            # Add technical indicator options
            indicator_options = [
                "1. All indicators (default)",
                "2. Basic indicators only (faster)"
            ]
            detail_indicator_choice = st.selectbox("Technical indicators", indicator_options, index=1)
            detail_indicator_choice = detail_indicator_choice.split(".")[0]
            
            # Add prediction target options
            target_options = [
                "1. Next day price direction (default)",
                "2. Next day return > threshold",
                "3. 3-day return direction",
                "4. 5-day return direction"
            ]
            detail_target_choice = st.selectbox("Prediction target", target_options, index=0)
            detail_target_choice = detail_target_choice.split(".")[0]
        
        if st.button("Analyze Stock", key='analyze_nifty_stock'):
        
            end_date = datetime.now().strftime("%Y-%m-%d")
            stock_model, comparison_df, performance, test_performance, best_model = analyze_single_stock(
                ticker=selected_stock,
                start_date=detail_start_date.strftime('%Y-%m-%d'),
                end_date=end_date,
                test_period=detail_test_period,
                investment=detail_investment,
                model_choice=detail_model_choice,
                indicator_choice=detail_indicator_choice,
                target_choice=detail_target_choice
            )
            
            if stock_model:
                st.session_state['nifty_stock_model'] = stock_model
                st.session_state['nifty_comparison_df'] = comparison_df
                st.session_state['nifty_performance'] = performance
                st.session_state['nifty_test_performance'] = test_performance
                st.session_state['nifty_best_model'] = best_model
                st.success(f"Analysis of {selected_stock_name} completed!")
            else:
                st.error(f"Analysis of {selected_stock_name} failed.")
        
        if 'nifty_stock_model' in st.session_state:
            stock_model = st.session_state['nifty_stock_model']
            comparison_df = st.session_state['nifty_comparison_df']
            performance = st.session_state['nifty_performance']
            test_performance = st.session_state['nifty_test_performance']
            best_model = st.session_state['nifty_best_model']
            
            # Show latest trading recommendation
            st.subheader("🚨 Latest Trading Recommendation")
            
            if 'ML_Trade' in stock_model.stock_data.columns:
                latest_trade = stock_model.stock_data.iloc[-1]
                latest_signal = 'BUY' if latest_trade['ML_Signal'] == 1 else 'SELL' if latest_trade['ML_Signal'] == 0 else 'HOLD' if latest_trade['ML_Signal'] == 0.5 else 'UNKNOWN'
                
                if latest_signal == 'BUY':
                    entry_price = latest_trade['Close']
                    target_price = latest_trade['Bollinger_Upper_20'] if 'Bollinger_Upper_20' in latest_trade else entry_price * 1.05
                    stop_loss_price = latest_trade['Bollinger_Lower_20'] if 'Bollinger_Lower_20' in latest_trade else entry_price * 0.95
                
                    st.markdown(f"""
                    ✅ **BULLISH SIGNAL - Enter LONG position**
                    - 🚀 Entry Price: ₹{entry_price:.2f}
                    - 🎯 Target Price: ₹{target_price:.2f}
                    - 🛑 Stop-Loss Price: ₹{stop_loss_price:.2f}
                    """)
                elif latest_signal == 'HOLD':
                    current_price = latest_trade['Close']
                    upper_range = latest_trade['Bollinger_Upper_20'] if 'Bollinger_Upper_20' in latest_trade else current_price * 1.03
                    lower_range = latest_trade['Bollinger_Lower_20'] if 'Bollinger_Lower_20' in latest_trade else current_price * 0.97
                    
                    st.markdown(f"""
                    🟡 **NEUTRAL SIGNAL - Maintain Current Positions**
                    - 💰 Current Price: ₹{current_price:.2f}
                    - 📈 Upper Range: ₹{upper_range:.2f}
                    - 📉 Lower Range: ₹{lower_range:.2f}
                    - ℹ️ Consider waiting for a stronger signal before making new trades
                    """)
                else:
                    exit_price = latest_trade['Close']
                    target_price = latest_trade['Bollinger_Lower_20'] if 'Bollinger_Lower_20' in latest_trade else exit_price * 0.95
                    stop_loss_price = latest_trade['Bollinger_Upper_20'] if 'Bollinger_Upper_20' in latest_trade else exit_price * 1.05
                
                    st.markdown(f"""
                    ⛔ **BEARISH SIGNAL - Exit LONG position or Enter SHORT position**
                    - 🚀 Exit/Short Price: ₹{exit_price:.2f}
                    - 🎯 Target Price: ₹{target_price:.2f}
                    - 🛑 Stop-Loss Price: ₹{stop_loss_price:.2f}
                    """)
            
            # Show stock information and analysis results
            st.subheader(f"📊 {selected_stock_name} Analysis Results")
            
            st.subheader("Best Model : " + best_model)
                
            col2, col3, col4 = st.columns(3)
            with col2:
                st.metric("Return", f"{performance['Strategy Return']:.2%}")
            with col3:
                st.metric("Profit/Loss", f"₹{performance['Profit/Loss']:,.2f}")
            with col4:
                st.metric("Sharpe Ratio", f"{performance['Strategy Sharpe']:.2f}")
                
            # Show test performance metrics
            st.subheader("📊 Test Period Performance")
            col2, col3, col4 = st.columns(3)
            with col2:
                st.metric("Test Return", f"{test_performance['Strategy Return']:.2%}")
            with col3:
                st.metric("Test Profit/Loss", f"₹{test_performance['Profit/Loss']:,.2f}")
            with col4:
                st.metric("Test Sharpe Ratio", f"{test_performance['Strategy Sharpe']:.2f}")
                
            # Show detailed metrics
            st.subheader("📈 Performance Metrics")
            
            metrics_data = {
                "Metric": [
                    "Initial Investment",
                    "Final Strategy Value",
                    "Strategy Return",
                    "Buy & Hold Return",
                    "Strategy Annual Return",
                    "Buy & Hold Annual Return",
                    "Strategy Sharpe",
                    "Buy & Hold Sharpe",
                    "Strategy Max Drawdown",
                    "Buy & Hold Max Drawdown"
                ],
                "TestPeriod": [
                    f"₹{test_performance['Initial Investment']:,.2f}",
                    f"₹{test_performance['Final Strategy Value']:,.2f}",
                    f"{test_performance['Strategy Return']:.2%}",
                    f"{test_performance['Buy & Hold Return']:.2%}",
                    f"{test_performance['Strategy Annual Return']:.2%}",
                    f"{test_performance['Buy & Hold Annual Return']:.2%}",
                    f"{test_performance['Strategy Sharpe']:.2f}",
                    f"{test_performance['Buy & Hold Sharpe']:.2f}",
                    f"{test_performance['Strategy Max Drawdown']:.2%}",
                    f"{test_performance['Buy & Hold Max Drawdown']:.2%}"
                ], 
                "History": [
                    f"₹{performance['Initial Investment']:,.2f}",
                    f"₹{performance['Final Strategy Value']:,.2f}",
                    f"{performance['Strategy Return']:.2%}",
                    f"{performance['Buy & Hold Return']:.2%}",
                    f"{performance['Strategy Annual Return']:.2%}",
                    f"{performance['Buy & Hold Annual Return']:.2%}",
                    f"{performance['Strategy Sharpe']:.2f}",
                    f"{performance['Buy & Hold Sharpe']:.2f}",
                    f"{performance['Strategy Max Drawdown']:.2%}",
                    f"{performance['Buy & Hold Max Drawdown']:.2%}"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)
            
            # Show model comparison
            st.subheader("🤖 Model Comparison")
            
            if not comparison_df.empty:
                # Format percentages for display
                display_df = comparison_df.copy()
                for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Plot model comparison
                st.subheader("📊 Model Performance Comparison")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                x = np.arange(len(comparison_df['Model']))
                width = 0.2
                
                for i, metric in enumerate(metrics):
                    ax.bar(x + i*width, comparison_df[metric], width, label=metric)
                
                ax.set_xlabel('Models')
                ax.set_ylabel('Score')
                ax.set_title(f'Model Comparison for {selected_stock_name} Stock Prediction')
                ax.set_xticks(x + width*1.5)
                ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            
            # Show trading signals and strategy performance
            st.subheader("📈 Trading Signals and Strategy Performance")
            
            if 'ML_Trade' in stock_model.stock_data.columns:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot 1: Price and Trading Signals
                ax1.plot(stock_model.stock_data.index, stock_model.stock_data['Close'], label='Close Price', alpha=0.5)
                
                # Filter out NaN values for plotting with three signal types
                buy_signals = stock_model.stock_data[stock_model.stock_data['ML_Trade'] == 1]
                hold_signals = stock_model.stock_data[stock_model.stock_data['ML_Trade'] == 0.5]
                sell_signals = stock_model.stock_data[stock_model.stock_data['ML_Trade'] == 0]
                
                ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', 
                           label=f'{best_model} Buy Signal', alpha=1, s=100)
                ax1.scatter(hold_signals.index, hold_signals['Close'], marker='o', color='gold', 
                           label=f'{best_model} Hold Signal', alpha=1, s=80)
                ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', 
                           label=f'{best_model} Sell Signal', alpha=1, s=100)
                
                ax1.set_title(f"{selected_stock_name} Trading Signals using {best_model}")
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price')
                ax1.legend()
                ax1.grid(True)            
                
                # Plot 2: Strategy Performance
                ax2.plot(stock_model.stock_data.index, stock_model.stock_data['Market_Value'], 
                        label=f'Buy & Hold (₹{stock_model.initial_investment:,})', color='blue')
                ax2.plot(stock_model.stock_data.index, stock_model.stock_data['Strategy_Value'], 
                        label=f'{best_model} Strategy (₹{stock_model.initial_investment:,})', color='green')
                ax2.set_title(f"{selected_stock_name} Strategy Performance Comparison")
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Portfolio Value (₹)')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show a table of recent signals
                st.subheader("🔍 Recent Trading Signals")
                
                recent_signals = stock_model.stock_data.tail(10)[['Close', 'ML_Signal', 'ML_Trade', 'Returns', 'Strategy_Return']].copy()
                recent_signals.index = recent_signals.index.strftime('%Y-%m-%d')
                recent_signals['Signal'] = recent_signals['ML_Signal'].apply(lambda x: 'BUY' if x == 1 else 'SELL' if x == 0 else 'HOLD' if x == 0.5 else 'UNKNOWN')                
                recent_signals = recent_signals[['Close', 'Signal', 'Returns', 'Strategy_Return']]
                recent_signals.columns = ['Close Price (₹)', 'Signal', 'Market Return (%)', 'Strategy Return (%)']
                recent_signals['Market Return (%)'] = recent_signals['Market Return (%)'].apply(lambda x: f"{x*100:.2f}%")
                recent_signals['Strategy Return (%)'] = recent_signals['Strategy Return (%)'].apply(lambda x: f"{x*100:.2f}%")
                st.table(recent_signals)
            
            # If the model is tree-based, show feature importance
            if best_model in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Random Forest_Tuned', 'Gradient Boosting_Tuned', 'XGBoost_Tuned']:
                st.subheader("🔍 Feature Importance")
                
                # Get feature importance
                try:
                    model = stock_model.results[best_model]['model']
                    
                    if 'XGBoost' in best_model:
                        importance = model.feature_importances_
                    else:
                        importance = model.feature_importances_
                        
                    # Sort features by importance
                    feature_names = stock_model.X.columns
                    indices = np.argsort(importance)[::-1]
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(len(indices[:15])), importance[indices[:15]], align='center')
                    ax.set_xticks(range(len(indices[:15])))
                    ax.set_xticklabels([feature_names[i] for i in indices[:15]], rotation=90)
                    ax.set_title(f'Feature Importance for {best_model}')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show top 10 features in a table
                    st.subheader("🔝 Top 10 Important Features")
                    
                    top_features = []
                    for i, idx in enumerate(indices[:10]):
                        top_features.append({
                            'Rank': i+1,
                            'Feature': feature_names[idx],
                            'Importance': f"{importance[idx]:.4f}"
                        })
                    
                    st.table(pd.DataFrame(top_features))
                except Exception as e:
                    st.error(f"Error showing feature importance: {str(e)}")
    
    with tab3:
        st.header("🌏 Custom Stock Analysis")
        st.markdown("Enter any stock ticker to perform detailed analysis.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, GOOGL)", "AAPL")

            custom_start_date = st.date_input("Select start date", 
                                        value=datetime.now() - timedelta(days=365*2),
                                        format="YYYY-MM-DD",
                                        key='custom_start_date')
            custom_test_period = st.slider("Test period (months)", 1, 12, 6, key='custom_test_period')

        
        with col2:
            custom_investment = st.number_input("Initial investment amount ($)", 
                                             min_value=1000, 
                                             max_value=1000000, 
                                             value=10000,
                                             step=1000,
                                             key='custom_investment')
            
            # Add model selection options
            model_options = [
                "1. All models (default)",
                "2. Traditional ML models only (faster)",
                "3. Deep learning models only"
            ]
            custom_model_choice = st.selectbox("Model selection", model_options, index=1, key='custom_model')
            custom_model_choice = custom_model_choice.split(".")[0]
            
            # Add technical indicator options
            indicator_options = [
                "1. All indicators (default)",
                "2. Basic indicators only (faster)"
            ]
            custom_indicator_choice = st.selectbox("Technical indicators", indicator_options, index=1, key='custom_indicator')
            custom_indicator_choice = custom_indicator_choice.split(".")[0]
            
            # Add prediction target options
            target_options = [
                "1. Next day price direction (default)",
                "2. Next day return > threshold",
                "3. 3-day return direction",
                "4. 5-day return direction"
            ]
            custom_target_choice = st.selectbox("Prediction target", target_options, index=0, key='custom_target')
            custom_target_choice = custom_target_choice.split(".")[0]
        
        if st.button("Analyze Custom Stock", key='analyze_custom_stock'):
            end_date = datetime.now().strftime("%Y-%m-%d")
            try:             
                stock_model, comparison_df, performance,test_performance,  best_model = analyze_single_stock(
                    ticker=custom_ticker,
                    start_date=custom_start_date.strftime('%Y-%m-%d'),
                    end_date=end_date,
                    test_period=custom_test_period,
                    investment=custom_investment,
                    model_choice=custom_model_choice,
                    indicator_choice=custom_indicator_choice,
                    target_choice=custom_target_choice
                )
                
                if stock_model:
                    st.session_state['custom_stock_model'] = stock_model
                    st.session_state['custom_comparison_df'] = comparison_df
                    st.session_state['custom_performance'] = performance
                    st.session_state['custom_test_performance'] = test_performance
                    
                    st.session_state['custom_best_model'] = best_model
                    st.success(f"Analysis of {custom_ticker} completed!")
                else:
                    st.error(f"Analysis of {custom_ticker} failed.")
            except Exception as e:
                st.error(f"Error analyzing {custom_ticker}: {str(e)}")
                
        
        if 'custom_stock_model' in st.session_state:
            stock_model = st.session_state['custom_stock_model']
            comparison_df = st.session_state['custom_comparison_df']
            performance = st.session_state['custom_performance']
            test_performance = st.session_state['custom_test_performance']
            best_model = st.session_state['custom_best_model']
            
            # Show latest trading recommendation
            st.subheader("🚨 Latest Trading Recommendation")
            
            
            if 'ML_Trade' in stock_model.stock_data.columns:
                
                latest_trade = stock_model.stock_data.iloc[-1]
                latest_signal = 'BUY' if latest_trade['ML_Signal'] == 1 else 'SELL' if latest_trade['ML_Signal'] == 0 else 'HOLD' if latest_trade['ML_Signal'] == 0.5 else 'UNKNOWN'
                
            # Update the latest trading recommendation logic
            
                if latest_signal == 'BUY':
                    entry_price = latest_trade['Close']
                    target_price = latest_trade['Bollinger_Upper_20'] if 'Bollinger_Upper_20' in latest_trade else entry_price * 1.05
                    stop_loss_price = latest_trade['Bollinger_Lower_20'] if 'Bollinger_Lower_20' in latest_trade else entry_price * 0.95
                
                    st.markdown(f"""
                    ✅ **BULLISH SIGNAL - Enter LONG position**
                    - 🚀 Entry Price: ${entry_price:.2f}
                    - 🎯 Target Price: ${target_price:.2f}
                    - 🛑 Stop-Loss Price: ${stop_loss_price:.2f}
                    """)
                elif latest_signal == 'HOLD':
                    current_price = latest_trade['Close']
                    upper_range = latest_trade['Bollinger_Upper_20'] if 'Bollinger_Upper_20' in latest_trade else current_price * 1.03
                    lower_range = latest_trade['Bollinger_Lower_20'] if 'Bollinger_Lower_20' in latest_trade else current_price * 0.97
                    
                    st.markdown(f"""
                    🟡 **NEUTRAL SIGNAL - Maintain Current Positions**
                    - 💰 Current Price: ${current_price:.2f}
                    - 📈 Upper Range: ${upper_range:.2f}
                    - 📉 Lower Range: ${lower_range:.2f}
                    - ℹ️ Consider waiting for a stronger signal before making new trades
                    """)
                else:
                    exit_price = latest_trade['Close']
                    target_price = latest_trade['Bollinger_Lower_20'] if 'Bollinger_Lower_20' in latest_trade else exit_price * 0.95
                    stop_loss_price = latest_trade['Bollinger_Upper_20'] if 'Bollinger_Upper_20' in latest_trade else exit_price * 1.05
                
                    st.markdown(f"""
                    ⛔ **BEARISH SIGNAL - Exit LONG position or Enter SHORT position**
                    - 🚀 Exit/Short Price: ${exit_price:.2f}
                    - 🎯 Target Price: ${target_price:.2f}
                    - 🛑 Stop-Loss Price: ${stop_loss_price:.2f}
                    """)
            
            # Show stock information and analysis results
            st.subheader(f"📊 {stock_model.ticker} Analysis Results")
            
            st.subheader("Best Model : " + best_model)
                
            col2, col3, col4 = st.columns(3)
            with col2:
                st.metric("Return", f"{performance['Strategy Return']:.2%}")
            with col3:
                st.metric("Profit/Loss", f"${performance['Profit/Loss']:,.2f}")
            with col4:
                st.metric("Sharpe Ratio", f"{performance['Strategy Sharpe']:.2f}")
                
            # Show main metrics
            col2, col3, col4 = st.columns(3)
            with col2:
                st.metric("Return", f"{test_performance['Strategy Return']:.2%}")
            with col3:
                st.metric("Profit/Loss", f"${test_performance['Profit/Loss']:,.2f}")
            with col4:
                st.metric("Sharpe Ratio", f"{test_performance['Strategy Sharpe']:.2f}")
                
                
            
            # Show detailed metrics
            st.subheader("📈 Performance Metrics")
            
            metrics_data = {
                "Metric": [
                    "Initial Investment",
                    "Final Strategy Value",
                    "Strategy Return",
                    "Buy & Hold Return",
                    "Strategy Annual Return",
                    "Buy & Hold Annual Return",
                    "Strategy Sharpe",
                    "Buy & Hold Sharpe",
                    "Strategy Max Drawdown",
                    "Buy & Hold Max Drawdown"
                ],
                "TestPeriod": [
                    f"${test_performance['Initial Investment']:,.2f}",
                    f"${test_performance['Final Strategy Value']:,.2f}",
                    f"{test_performance['Strategy Return']:.2%}",
                    f"{test_performance['Buy & Hold Return']:.2%}",
                    f"{test_performance['Strategy Annual Return']:.2%}",
                    f"{test_performance['Buy & Hold Annual Return']:.2%}",
                    f"{test_performance['Strategy Sharpe']:.2f}",
                    f"{test_performance['Buy & Hold Sharpe']:.2f}",
                    f"{test_performance['Strategy Max Drawdown']:.2%}",
                    f"{test_performance['Buy & Hold Max Drawdown']:.2%}"
                ], 
                "History": [
                    f"${performance['Initial Investment']:,.2f}",
                    f"${performance['Final Strategy Value']:,.2f}",
                    f"{performance['Strategy Return']:.2%}",
                    f"{performance['Buy & Hold Return']:.2%}",
                    f"{performance['Strategy Annual Return']:.2%}",
                    f"{performance['Buy & Hold Annual Return']:.2%}",
                    f"{performance['Strategy Sharpe']:.2f}",
                    f"{performance['Buy & Hold Sharpe']:.2f}",
                    f"{performance['Strategy Max Drawdown']:.2%}",
                    f"{performance['Buy & Hold Max Drawdown']:.2%}"
                ] 
                
                
                
                
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)
            
            # Show model comparison
            st.subheader("🤖 Model Comparison")
            
            if not comparison_df.empty:
                # Format percentages for display
                display_df = comparison_df.copy()
                for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Plot model comparison
                st.subheader("📊 Model Performance Comparison")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                x = np.arange(len(comparison_df['Model']))
                width = 0.2
                
                for i, metric in enumerate(metrics):
                    ax.bar(x + i*width, comparison_df[metric], width, label=metric)
                
                ax.set_xlabel('Models')
                ax.set_ylabel('Score')
                ax.set_title(f'Model Comparison for {stock_model.ticker} Stock Prediction')
                ax.set_xticks(x + width*1.5)
                ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            
            # Show trading signals and strategy performance
            st.subheader("📈 Trading Signals and Strategy Performance")
            
            if 'ML_Trade' in stock_model.stock_data.columns:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot 1: Price and Trading Signals
                ax1.plot(stock_model.stock_data.index, stock_model.stock_data['Close'], label='Close Price', alpha=0.5)
                
                # Filter out NaN values for plotting with three signal types
                buy_signals = stock_model.stock_data[stock_model.stock_data['ML_Trade'] == 1]
                hold_signals = stock_model.stock_data[stock_model.stock_data['ML_Trade'] == 0.5]
                sell_signals = stock_model.stock_data[stock_model.stock_data['ML_Trade'] == 0]
                
                ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', 
                          label=f'{best_model} Buy Signal', alpha=1, s=100)
                ax1.scatter(hold_signals.index, hold_signals['Close'], marker='o', color='gold', 
                          label=f'{best_model} Hold Signal', alpha=1, s=80)
                ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', 
                          label=f'{best_model} Sell Signal', alpha=1, s=100)
                
                ax1.set_title(f"{stock_model.ticker} Trading Signals using {best_model}")
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price')
                ax1.legend()
                ax1.grid(True)            
                
                # Plot 2: Strategy Performance
                ax2.plot(stock_model.stock_data.index, stock_model.stock_data['Market_Value'], 
                       label=f'Buy & Hold (${stock_model.initial_investment:,})', color='blue')
                ax2.plot(stock_model.stock_data.index, stock_model.stock_data['Strategy_Value'], 
                       label=f'{best_model} Strategy (${stock_model.initial_investment:,})', color='green')
                ax2.set_title(f"{stock_model.ticker} Strategy Performance Comparison")
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Portfolio Value ($)')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show a table of recent signals
                st.subheader("🔍 Recent Trading Signals")
                
                recent_signals = stock_model.stock_data.tail(10)[['Close', 'ML_Signal', 'ML_Trade', 'Returns', 'Strategy_Return']].copy()
                recent_signals.index = recent_signals.index.strftime('%Y-%m-%d')
                recent_signals['Signal'] = recent_signals['ML_Signal'].apply( lambda x: 'BUY' if x == 1 else 'SELL' if x == 0 else 'HOLD' if x == 0.5 else 'UNKNOWN')                
                recent_signals = recent_signals[['Close', 'Signal', 'Returns', 'Strategy_Return']]
                recent_signals.columns = ['Close Price ($)', 'Signal', 'Market Return (%)', 'Strategy Return (%)']
                recent_signals['Market Return (%)'] = recent_signals['Market Return (%)'].apply(lambda x: f"{x*100:.2f}%")
                recent_signals['Strategy Return (%)'] = recent_signals['Strategy Return (%)'].apply(lambda x: f"{x*100:.2f}%")
                st.table(recent_signals)
                
            
            # If the model is tree-based, show feature importance
            if best_model in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Random Forest_Tuned', 'Gradient Boosting_Tuned', 'XGBoost_Tuned']:
                st.subheader("🔍 Feature Importance")
                
                # Get feature importance
                try:
                    model = stock_model.results[best_model]['model']
                    
                    if 'XGBoost' in best_model:
                        importance = model.feature_importances_
                    else:
                        importance = model.feature_importances_
                        
                    # Sort features by importance
                    feature_names = stock_model.X.columns
                    indices = np.argsort(importance)[::-1]
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(len(indices[:15])), importance[indices[:15]], align='center')
                    ax.set_xticks(range(len(indices[:15])))
                    ax.set_xticklabels([feature_names[i] for i in indices[:15]], rotation=90)
                    ax.set_title(f'Feature Importance for {best_model}')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show top 10 features in a table
                    st.subheader("🔝 Top 10 Important Features")
                    
                    top_features = []
                    for i, idx in enumerate(indices[:10]):
                        top_features.append({
                            'Rank': i+1,
                            'Feature': feature_names[idx],
                            'Importance': f"{importance[idx]:.4f}"
                        })
                    
                    st.table(pd.DataFrame(top_features))
                except Exception as e:
                    st.error(f"Error showing feature importance: {str(e)}")
    
    with tab4:
        portfolio_analyzer_tab()

# Add a function to StockModelComparison class to estimate returns
def add_estimate_returns_function():
    """Add a function to estimate returns for the StockModelComparison class"""
    return """
    def estimate_test_returns(self):
        \"\"\"Calculate test period performance metrics\"\"\"
        if 'ML_Trade' not in self.stock_data.columns:
            print("⚠️ No trading signals generated. Please run generate_trading_signals() first.")
            return None
            
        # Get the test period data
        test_data = self.stock_data.iloc[-int(self.test_period_months * 21):]  # Approx 21 trading days per month
        
        if len(test_data) == 0:
            print("⚠️ Test period contains no data.")
            return None
        
        # Calculate cumulative returns
        market_return = test_data['Cumulative_Market_Return'].iloc[-1] / test_data['Cumulative_Market_Return'].iloc[0] - 1
        strategy_return = test_data['Cumulative_Strategy_Return'].iloc[-1] / test_data['Cumulative_Strategy_Return'].iloc[0] - 1
        
        # Calculate alpha (excess return over market)
        alpha = strategy_return - market_return
        
        # Calculate volatility
        market_volatility = test_data['Returns'].std() * np.sqrt(252)  # Annualize
        strategy_volatility = test_data['Strategy_Return'].std() * np.sqrt(252)  # Annualize
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate for simplicity)
        market_sharpe = (market_return / len(test_data) * 252) / market_volatility if market_volatility > 0 else 0
        strategy_sharpe = (strategy_return / len(test_data) * 252) / strategy_volatility if strategy_volatility > 0 else 0
        
        # Calculate win rate
        win_rate = (test_data['Strategy_Return'] > 0).sum() / len(test_data[~test_data['Strategy_Return'].isna()])
        
        # Calculate profit factor (sum of positive returns / abs sum of negative returns)
        positive_returns = test_data['Strategy_Return'][test_data['Strategy_Return'] > 0].sum()
        negative_returns = abs(test_data['Strategy_Return'][test_data['Strategy_Return'] < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
        # Annualize returns for longer-term comparison
        days = len(test_data)
        annual_factor = 252 / days
        strategy_annual_return = (1 + strategy_return) ** annual_factor - 1
        market_annual_return = (1 + market_return) ** annual_factor - 1
        
        return {
            'Strategy Return': strategy_return,
            'Market Return': market_return,
            'Alpha': alpha,
            'Strategy Volatility': strategy_volatility,
            'Market Volatility': market_volatility,
            'Strategy Sharpe': strategy_sharpe,
            'Market Sharpe': market_sharpe,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Strategy Annual Return': strategy_annual_return,
            'Market Annual Return': market_annual_return,
            'Test Period Days': days
        }
    """ 

if __name__ == "__main__":
    # Try to modify StockModelComparison class
    try:
        from StockPrediction import StockModelComparison
        # Check if the estimate_test_returns method already exists
        if not hasattr(StockModelComparison, 'estimate_test_returns'):
            # Add the method
            import inspect
            import types
            
            # Create the function
            exec_globals = {}
            exec(add_estimate_returns_function(), exec_globals)
            estimate_test_returns = exec_globals['estimate_test_returns']
            
            # Add the method to the class
            StockModelComparison.estimate_test_returns = types.MethodType(estimate_test_returns, StockModelComparison)
    except Exception as e:
        st.error(f"Error modifying StockModelComparison class: {str(e)}")
        st.info("Please add the estimate_test_returns method to your StockPrediction.py file.")

    main()