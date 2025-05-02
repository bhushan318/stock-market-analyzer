# -*- coding: utf-8 -*-
"""
Created on Thu May  1 19:23:26 2025

@author: NagabhushanamTattaga
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime # timedelta
from sklearn.preprocessing import MinMaxScaler #, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split #TimeSeriesSplit, , GridSearchCV  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Conv1D, MaxPooling1D #, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️ pmdarima not installed. ARIMA forecasting will be disabled.")

class StockModelComparison:
    def __init__(self, ticker="IOVA", start_date="2018-01-01", end_date=None, 
                test_period_months=6, initial_investment=10000):
        """Initialize with stock ticker and parameters"""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        self.test_period_months = test_period_months
        self.initial_investment = initial_investment
        
        # Initialize data containers
        self.stock_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Initialize model and results containers
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.arima_model = None
        self.ensemble_model = None
        self.hyperparams = {}
        
        # Configuration settings
        self.enabled_models = "all"  # all, traditional, deep_learning, or list of specific models
        self.indicator_level = "all"  # all, basic, custom
        self.target_type = "next_day_direction"  # Options: next_day_direction, threshold, multi_day
        self.target_params = {"threshold": 0.01, "days": 1}
        
    def set_enabled_models(self, choice):
        """Set which models to enable based on user choice"""
        if choice == "1" or choice.lower() == "all":
            self.enabled_models = "all"
        elif choice == "2" or choice.lower() == "traditional":
            self.enabled_models = "traditional"
        elif choice == "3" or choice.lower() == "deep_learning":
            self.enabled_models = "deep_learning"
        elif choice == "4" or choice.lower() == "custom":
            # For custom model selection, pass a list of model names
            print("Select specific models:")
            print("1. Random Forest")
            print("2. Gradient Boosting")
            print("3. XGBoost")
            print("4. SVM")
            print("5. MLP (Neural Network)")
            print("6. AdaBoost")
            print("7. Logistic Regression")
            print("8. Ensemble")
            print("9. LSTM")
            print("10. CNN-LSTM")
            print("11. Bidirectional GRU")
            
            selected = input("Enter model numbers separated by commas: ")
            model_map = {
                "1": "Random Forest", "2": "Gradient Boosting", "3": "XGBoost",
                "4": "SVM", "5": "MLP", "6": "AdaBoost", "7": "Logistic Regression",
                "8": "Ensemble", "9": "LSTM", "10": "CNN-LSTM", "11": "Bidirectional GRU"
            }
            
            self.enabled_models = [model_map[x.strip()] for x in selected.split(",") if x.strip() in model_map]
        return self.enabled_models
    
    def set_indicator_level(self, choice):
        """Set the level of technical indicators to calculate"""
        if choice == "1" or choice.lower() == "all":
            self.indicator_level = "all"
        elif choice == "2" or choice.lower() == "basic":
            self.indicator_level = "basic"
        elif choice == "3" or choice.lower() == "custom":
            # For future implementation: allow custom indicator selection
            pass
        return self.indicator_level
    
    def set_target_type(self, choice):
        """Set the prediction target type"""
        if choice == "1":
            self.target_type = "next_day_direction"
        elif choice == "2":
            self.target_type = "threshold"
            threshold = input("Enter return threshold percentage (default: 1%): ") or "1"
            self.target_params["threshold"] = float(threshold) / 100
        elif choice == "3" or choice == "4":
            self.target_type = "multi_day"
            days = 3 if choice == "3" else 5
            self.target_params["days"] = days
        return self.target_type
        
    def download_data(self):
        """Download stock data and prepare initial DataFrame"""
        print(f"📥 Downloading data for {self.ticker}...")
        # For Indian stocks, try alternative methods
        if self.ticker.endswith('.NS'):
            # Try without the .NS extension
            base_ticker = self.ticker.replace('.NS', '')
            self.stock_data = yf.download(f"{base_ticker}.NS", start=self.start_date, end=self.end_date)
            
            # If still no data, try with BSE extension
            if len(self.stock_data) == 0:
                self.stock_data = yf.download(f"{base_ticker}.BO", start=self.start_date, end=self.end_date)
        
        # If still no data, raise error
        if len(self.stock_data) == 0:
            raise ValueError(f"No data found for ticker {self.ticker}. Please check the ticker symbol.")

    # self.stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        if len(self.stock_data) == 0:
            raise ValueError(f"No data found for ticker {self.ticker}. Please check the ticker symbol.")
            
        self.stock_data.columns = [i[0] for i in self.stock_data.columns]
        print(f"✅ Downloaded {len(self.stock_data)} days of data from {self.stock_data.index[0].strftime('%Y-%m-%d')} to {self.stock_data.index[-1].strftime('%Y-%m-%d')}")
        return self.stock_data
    
    def calculate_technical_indicators(self):
        """Calculate various technical indicators for feature engineering"""
        print("📊 Calculating technical indicators...")
        
        # Price and Volume indicators
        self.stock_data['Returns'] = self.stock_data['Close'].pct_change()
        self.stock_data['Log_Returns'] = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1))
        self.stock_data['Volume_Change'] = self.stock_data['Volume'].pct_change()
        self.stock_data['Volume_MA_20'] = self.stock_data['Volume'].rolling(window=20).mean()
        self.stock_data['Volume_Ratio'] = self.stock_data['Volume'] / self.stock_data['Volume_MA_20']
        
        # Basic indicators (always calculated)
        if self.indicator_level in ["all", "basic", "custom"]:
            # Moving averages
            for window in [20, 50, 200]:
                self.stock_data[f'SMA_{window}'] = self.stock_data['Close'].rolling(window=window).mean()
                self.stock_data[f'EMA_{window}'] = self.stock_data['Close'].ewm(span=window, adjust=False).mean()
            
            # RSI - 14 period
            delta = self.stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            self.stock_data['RSI_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            self.stock_data['MACD'] = self.stock_data['Close'].ewm(span=12, adjust=False).mean() - \
                                     self.stock_data['Close'].ewm(span=26, adjust=False).mean()
            self.stock_data['MACD_Signal'] = self.stock_data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            mid = self.stock_data['Close'].rolling(window=20).mean()
            std = self.stock_data['Close'].rolling(window=20).std()
            self.stock_data['Bollinger_Upper_20'] = mid + 2 * std
            self.stock_data['Bollinger_Lower_20'] = mid - 2 * std
        
        # Advanced indicators (calculated for 'all' or 'custom' levels)
        if self.indicator_level in ["all", "custom"]:
            # Additional trend indicators
            for window in [5, 10, 20, 50, 100, 200]:
                if window not in [20, 50, 200]:  # Skip those already calculated
                    self.stock_data[f'SMA_{window}'] = self.stock_data['Close'].rolling(window=window).mean()
                    self.stock_data[f'EMA_{window}'] = self.stock_data['Close'].ewm(span=window, adjust=False).mean()
            
            # Price distance from moving averages (normalized)
            for window in [20, 50, 200]:
                self.stock_data[f'Price_to_SMA_{window}'] = self.stock_data['Close'] / self.stock_data[f'SMA_{window}'] - 1
            
            # Moving average crossovers
            self.stock_data['SMA_5_10_Cross'] = np.where(self.stock_data['SMA_5'] > self.stock_data['SMA_10'], 1, -1)
            self.stock_data['SMA_10_20_Cross'] = np.where(self.stock_data['SMA_10'] > self.stock_data['SMA_20'], 1, -1)
            self.stock_data['SMA_50_200_Cross'] = np.where(self.stock_data['SMA_50'] > self.stock_data['SMA_200'], 1, -1)
            
            # Volatility indicators
            # True Range & ATR
            self.stock_data["High-Low"] = self.stock_data["High"] - self.stock_data["Low"]
            self.stock_data["High-Close"] = abs(self.stock_data["High"] - self.stock_data["Close"].shift(1))
            self.stock_data["Low-Close"] = abs(self.stock_data["Low"] - self.stock_data["Close"].shift(1))
            self.stock_data["True_Range"] = self.stock_data[["High-Low", "High-Close", "Low-Close"]].max(axis=1)
            
            for window in [5, 14, 21]:
                self.stock_data[f'ATR_{window}'] = self.stock_data["True_Range"].rolling(window=window).mean()
                self.stock_data[f'Volatility_{window}'] = self.stock_data['Returns'].rolling(window=window).std()
            
            # Additional Bollinger Band metrics
            for window in [20]:
                mid = self.stock_data['Close'].rolling(window=window).mean()
                std = self.stock_data['Close'].rolling(window=window).std()
                self.stock_data[f'Bollinger_Width_{window}'] = (mid + 2 * std - (mid - 2 * std)) / mid
                self.stock_data[f'Bollinger_%B_{window}'] = (self.stock_data['Close'] - (mid - 2 * std)) / (4 * std)
            
            # Momentum indicators
            # Additional RSI periods
            for window in [6, 21]:
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                rs = avg_gain / avg_loss
                self.stock_data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            # MACD Histogram
            self.stock_data['MACD_Hist'] = self.stock_data['MACD'] - self.stock_data['MACD_Signal']
            
            # Stochastic Oscillator
            for window in [14]:
                low_min = self.stock_data['Low'].rolling(window=window).min()
                high_max = self.stock_data['High'].rolling(window=window).max()
                self.stock_data[f'%K_{window}'] = 100 * ((self.stock_data['Close'] - low_min) / (high_max - low_min))
                self.stock_data[f'%D_{window}'] = self.stock_data[f'%K_{window}'].rolling(window=3).mean()
            
            # Implementation of common technical indicators without requiring external libs
            
            # Momentum
            for window in [10, 30]:
                self.stock_data[f'MOM_{window}'] = self.stock_data['Close'].diff(window)
            
            # ROC - Rate of Change
            for window in [10, 20]:
                self.stock_data[f'ROC_{window}'] = self.stock_data['Close'].pct_change(window) * 100
                
            # Williams %R
            for window in [14]:
                highest_high = self.stock_data['High'].rolling(window=window).max()
                lowest_low = self.stock_data['Low'].rolling(window=window).min()
                self.stock_data[f'WILLR_{window}'] = -100 * (highest_high - self.stock_data['Close']) / (highest_high - lowest_low)
            
            # OBV - On Balance Volume (simplified implementation)
            self.stock_data['OBV'] = 0
            self.stock_data.loc[self.stock_data['Close'] > self.stock_data['Close'].shift(1), 'OBV'] = self.stock_data['Volume']
            self.stock_data.loc[self.stock_data['Close'] < self.stock_data['Close'].shift(1), 'OBV'] = -self.stock_data['Volume']
            self.stock_data['OBV'] = self.stock_data['OBV'].cumsum()
            
            # Advanced patterns and indicators
            # Price patterns
            self.stock_data['Higher_Highs'] = np.where(
                self.stock_data['High'] > self.stock_data['High'].shift(1), 1, 0)
            self.stock_data['Lower_Lows'] = np.where(
                self.stock_data['Low'] < self.stock_data['Low'].shift(1), 1, 0)
            
            # Calculate higher highs and lower lows streaks
            self.stock_data['HH_Streak'] = self.stock_data['Higher_Highs'].rolling(window=5).sum()
            self.stock_data['LL_Streak'] = self.stock_data['Lower_Lows'].rolling(window=5).sum()
            
            # Chaikin Money Flow
            period = 20
            money_flow_multiplier = ((self.stock_data['Close'] - self.stock_data['Low']) - 
                                    (self.stock_data['High'] - self.stock_data['Close'])) / (self.stock_data['High'] - self.stock_data['Low'])
            money_flow_volume = money_flow_multiplier * self.stock_data['Volume']
            self.stock_data['CMF'] = money_flow_volume.rolling(window=period).sum() / self.stock_data['Volume'].rolling(window=period).sum()
            
            # Force Index
            self.stock_data['Force_Index_13'] = self.stock_data['Close'].diff(1) * self.stock_data['Volume']
            self.stock_data['Force_Index_13'] = self.stock_data['Force_Index_13'].ewm(span=13, adjust=False).mean()
            
            # Correlation between price and volume
            self.stock_data['Price_Volume_Corr'] = self.stock_data['Close'].rolling(window=20).corr(self.stock_data['Volume'])
            
            # Multiple timeframe features
            for window in [5, 15, 30]:
                # Compute the rate of change
                self.stock_data[f'Rate_Of_Change_{window}'] = self.stock_data['Close'].pct_change(periods=window)
                
                # Compute the z-score of close
                self.stock_data[f'Z_Score_{window}'] = (self.stock_data['Close'] - 
                                                      self.stock_data['Close'].rolling(window=window).mean()) / \
                                                      self.stock_data['Close'].rolling(window=window).std()
            
            # Compute trend duration
            self.stock_data['Trend_Up'] = np.where(self.stock_data['Close'] > self.stock_data['Close'].shift(1), 1, 0)
            self.stock_data['Up_Streak'] = 0
            
            # Calculate streak lengths
            streak = 0
            for i in range(1, len(self.stock_data)):
                if self.stock_data['Trend_Up'].iloc[i] == 1:
                    streak += 1
                else:
                    streak = 0
                self.stock_data['Up_Streak'].iloc[i] = streak
        
        # Create target variable based on user selection
        if self.target_type == "next_day_direction":
            # Predict next day's direction (up or down)
            self.stock_data['Next_Day_Change'] = self.stock_data['Close'].pct_change().shift(-1)
            self.stock_data['Target'] = np.where(self.stock_data['Next_Day_Change'] > 0, 1, 0)
            # print(self.stock_data[['Close', 'Next_Day_Change', 'Target']])
            
        elif self.target_type == "threshold":
            # Predict if return exceeds threshold
            threshold = self.target_params["threshold"]
            self.stock_data['Next_Day_Change'] = self.stock_data['Close'].pct_change().shift(-1)
            self.stock_data['Target'] = np.where(self.stock_data['Next_Day_Change'] > threshold, 1, 0)
            
        elif self.target_type == "multi_day":
            # Predict n-day forward returns
            days = self.target_params["days"]
            self.stock_data[f'Next_{days}_Day_Change'] = self.stock_data['Close'].pct_change(periods=days).shift(-days)
            self.stock_data['Target'] = np.where(self.stock_data[f'Next_{days}_Day_Change'] > 0, 1, 0)
        
        # Drop NaN values
        self.stock_data.dropna(inplace=True)
        print(f"✅ Calculated {len(self.stock_data.columns) - 7} technical indicators")
        
        return self.stock_data
    
    def prepare_features(self, test_size=None):
        """Prepare features and target variables"""
        print("🔄 Preparing features and splitting data...")
        
        # Select features (exclude date, price columns, and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj close', 'Volume', 'Target', 'Returns', 'Log_Returns']
        
        # Also exclude any columns containing 'Next' or 'Change' as they're target-related
        for col in self.stock_data.columns:
            if 'Next' in col or 'Change' in col:
                exclude_cols.append(col)
                
        feature_cols = [col for col in self.stock_data.columns if col not in exclude_cols]
        
        self.X = self.stock_data[feature_cols]
        self.y = self.stock_data['Target']
        
        # Scale features for better model performance
        scaler = MinMaxScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        # Calculate test size based on months if not provided
        if test_size is None:
            # Calculate number of trading days in test period (approx 21 trading days per month)
            test_days = self.test_period_months * 21
            if test_days >= len(self.X_scaled):
                print(f"⚠️ Warning: Requested test period ({test_days} days) is longer than available data ({len(self.X_scaled)} days)")
                test_days = int(len(self.X_scaled) * 0.2)  # Default to 20% if requested period is too long
            
            test_size = test_days / len(self.X_scaled)
            print(f"✅ Using last {test_days} days (approx. {self.test_period_months} months) for testing")
        
        # Split data - use time series split to avoid look-ahead bias
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, shuffle=False
        )
        
        print(f"✅ Data prepared: {len(self.X_train)} training samples, {len(self.X_test)} testing samples")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def prepare_lstm_data(self, sequence_length=60):
        """Prepare sequential data for LSTM model"""
        print(f"🔄 Preparing sequential data with {sequence_length} day lookback...")
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.X)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(scaled_data)):
            X_seq.append(scaled_data[i-sequence_length:i])
            y_seq.append(self.y.iloc[i])
            
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        # Calculate test size based on months
        test_days = self.test_period_months * 21
        if test_days >= len(X_seq):
            test_days = int(len(X_seq) * 0.2)  # Default to 20% if requested period is too long
        
        # Split into train and test sets
        train_size = len(X_seq) - test_days
        X_train_seq = X_seq[:train_size]
        y_train_seq = y_seq[:train_size]
        X_test_seq = X_seq[train_size:]
        y_test_seq = y_seq[train_size:]
        
        print(f"✅ LSTM data prepared: {X_train_seq.shape}, {y_train_seq.shape}")
        return X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler
        
    def build_models(self):
        """Build various ML models"""
        print("🔨 Building models...")
        
        # Determine which models to build based on user selection
        model_list = []
        
        if self.enabled_models == "all" or self.enabled_models == "traditional" or isinstance(self.enabled_models, list):
            # Traditional ML models
            trad_models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": xgb.XGBClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
            }
            
            # Filter models based on user selection
            if isinstance(self.enabled_models, list):
                trad_models = {k: v for k, v in trad_models.items() if k in self.enabled_models}
            
            # Add to model list
            self.models.update(trad_models)
            model_list.extend(list(trad_models.keys()))
            
            # Add ensemble model (only if we have multiple traditional models)
            if len(trad_models) > 1 and (self.enabled_models == "all" or self.enabled_models == "traditional" or "Ensemble" in self.enabled_models):
                estimators = [(k.lower().replace(' ', '_'), v) for k, v in trad_models.items()]
                self.models['Ensemble'] = VotingClassifier(estimators=estimators, voting='soft')
                model_list.append("Ensemble")
        
        print(f"✅ Built {len(self.models)} ML models: {', '.join(model_list)}")
        return self.models
        


    
    def train_evaluate_models(self):
        """Train and evaluate all ML models with standardized evaluation"""
        print("🏋️‍♂️ Training and evaluating models...")
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Evaluate model with standardized method
            metrics, _ = self.evaluate_model_performance(
                model=model, 
                X_data=self.X_scaled,  # Use all data to generate full signals
                y_data=self.y_test,    # But test metrics on test set
                is_sequence_model=False
            )
            
            # Store model and metrics
            self.results[name] = {
                'model': model,
                **metrics  # Unpack all metrics
            }
            
            # Store predictions
            y_pred = model.predict(self.X_scaled)
            self.results[name]['test_predictions'] = y_pred  # Use test_predictions key consistently
            
            print(f"    {name} - Accuracy: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1_score', 0):.4f}")
            print(f"    Train Return: {metrics.get('train_return', 0):.2%}, Test Return: {metrics.get('test_return', 0):.2%}")
        
        # Find best model by composite score if available, otherwise fall back to F1
        if self.results:
            if all('composite_score' in result for result in self.results.values()):
                best_model_name = max(self.results.items(), key=lambda x: x[1]['composite_score'])[0]
                print(f"✅ Best model (by composite score): {best_model_name}")
            else:
                best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
                print(f"✅ Best model (by F1 score): {best_model_name}")
            
            self.best_model_name = best_model_name
            self.best_model = self.results[best_model_name]['model']
            
            # Print best model details
            best_metrics = self.results[best_model_name]
            print(f"   F1 Score: {best_metrics.get('f1_score', 0):.4f}")
            print(f"   Train Return: {best_metrics.get('train_return', 0):.2%}")
            print(f"   Test Return: {best_metrics.get('test_return', 0):.2%}")
            if 'composite_score' in best_metrics:
                print(f"   Composite Score: {best_metrics['composite_score']:.4f}")
        else:
            print("⚠️ No models were trained and evaluated")
            
        return self.results


    def train_lstm_model(self, units=50, epochs=50):
        """Build and train LSTM model with standardized evaluation"""
        if self.enabled_models != "all" and self.enabled_models != "deep_learning" and "LSTM" not in self.enabled_models:
            print("⚠️ LSTM model disabled by user configuration")
            return None
                
        print("🧠 Building and training LSTM model...")
        
        # Prepare sequential data
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler = self.prepare_lstm_data()
        
        # Build LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=units, return_sequences=True, 
                           input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=units))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = lstm_model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate model with standardized method
        metrics, _ = self.evaluate_model_performance(
            model=lstm_model,
            X_data=None,  # Will use all data
            y_data=y_test_seq,
            is_sequence_model=True
        )
        
        # Get predictions on test data for storing
        y_pred_proba = lstm_model.predict(X_test_seq)
        
        # Apply 3-way classification
        y_pred = np.zeros(len(y_pred_proba))
        for i, prob in enumerate(y_pred_proba):
            if prob > 0.65:  # Strong confidence for BUY
                y_pred[i] = 1    # BUY signal
            elif prob < 0.35:  # Strong confidence for SELL
                y_pred[i] = 0    # SELL signal
            else:
                y_pred[i] = 0.5  # HOLD signal (uncertainty zone)
        
        # Store results
        self.results['LSTM'] = {
            'model': lstm_model,
            'history': history,
            'X_test_seq': X_test_seq,
            'y_test_seq': y_test_seq,
            'test_predictions': y_pred,  # Use test_predictions consistently
            **metrics  # Unpack all metrics
        }
        
        print(f"✅ LSTM model - Accuracy: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1_score', 0):.4f}")
        print(f"   Train Return: {metrics.get('train_return', 0):.2%}, Test Return: {metrics.get('test_return', 0):.2%}")
        
        # Update best model based on composite score if available
        if 'composite_score' in metrics:
            best_score = self.results.get(self.best_model_name, {}).get('composite_score', 0)
            if metrics['composite_score'] > best_score:
                self.best_model_name = 'LSTM'
                self.best_model = lstm_model
                print(f"🏆 LSTM is now the best model with composite score: {metrics['composite_score']:.4f}!")
        else:
            # Fall back to F1 score if composite is not available
            if metrics.get('f1_score', 0) > self.results.get(self.best_model_name, {}).get('f1_score', 0):
                self.best_model_name = 'LSTM'
                self.best_model = lstm_model
                print(f"🏆 LSTM is now the best model with F1: {metrics['f1_score']:.4f}!")
                
        return self.results['LSTM']




    def build_bidirectional_gru_model(self):
        """Build Bidirectional GRU model with standardized evaluation"""
        if self.enabled_models != "all" and self.enabled_models != "deep_learning" and "Bidirectional GRU" not in self.enabled_models:
            print("⚠️ Bidirectional GRU model disabled by user configuration")
            return None
                
        print("🧠 Building Bidirectional GRU model...")
        
        # Prepare sequence data
        X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler = self.prepare_lstm_data()
        
        # Define input shape
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        
        # Build Bidirectional GRU model
        model = Sequential()
        
        # Bidirectional GRU layers
        model.add(Bidirectional(GRU(units=64, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Bidirectional(GRU(units=32)))
        model.add(Dropout(0.3))
        
        # Output layers
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate model with standardized method
        metrics, _ = self.evaluate_model_performance(
            model=model,
            X_data=None,  # Will use all data
            y_data=y_test_seq,
            is_sequence_model=True
        )
        
        # Get predictions on test data for storing
        y_pred_proba = model.predict(X_test_seq)
        
        # Apply 3-way classification
        y_pred = np.zeros(len(y_pred_proba))
        for i, prob in enumerate(y_pred_proba):
            if prob > 0.65:  # Strong confidence for BUY
                y_pred[i] = 1    # BUY signal
            elif prob < 0.35:  # Strong confidence for SELL
                y_pred[i] = 0    # SELL signal
            else:
                y_pred[i] = 0.5  # HOLD signal (uncertainty zone)
        
        # Store results
        self.results['Bidirectional GRU'] = {
            'model': model,
            'history': history,
            'X_test_seq': X_test_seq,
            'y_test_seq': y_test_seq,
            'test_predictions': y_pred,  # Store as test_predictions instead of predictions
            **metrics  # Unpack all metrics
        }
        
        print(f"✅ Bidirectional GRU model - Accuracy: {metrics.get('accuracy', 0):.4f}, F1: {metrics.get('f1_score', 0):.4f}")
        print(f"   Train Return: {metrics.get('train_return', 0):.2%}, Test Return: {metrics.get('test_return', 0):.2%}")
        
        # Update best model based on composite score if available
        if 'composite_score' in metrics:
            best_score = self.results.get(self.best_model_name, {}).get('composite_score', 0)
            if metrics['composite_score'] > best_score:
                self.best_model_name = 'Bidirectional GRU'
                self.best_model = model
                print(f"🏆 Bidirectional GRU is now the best model with composite score: {metrics['composite_score']:.4f}!")
        else:
            # Fall back to F1 score if composite is not available
            if metrics.get('f1_score', 0) > self.results.get(self.best_model_name, {}).get('f1_score', 0):
                self.best_model_name = 'Bidirectional GRU'
                self.best_model = model
                print(f"🏆 Bidirectional GRU is now the best model with F1: {metrics['f1_score']:.4f}!")
                
        return self.results['Bidirectional GRU']

    def build_arima_model(self):
        """Build and fit ARIMA model for time series forecasting"""
        if not ARIMA_AVAILABLE:
            print("⚠️ ARIMA forecasting unavailable. Install pmdarima package to enable this feature.")
            return None
            
        print("🔄 Building ARIMA model for time series forecasting...")
        
        # Prepare time series data
        ts_data = self.stock_data['Close'].copy()
        
        # Auto ARIMA to find optimal parameters
        print("   Finding optimal ARIMA parameters (this may take a while)...")
        try:
            arima_model = auto_arima(
                ts_data,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                d=None, max_d=2,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            order = arima_model.order
            print(f"✅ Optimal ARIMA order: {order}")
            
            # Fit final ARIMA model
            final_model = SARIMAX(ts_data, order=order)
            self.arima_model = final_model.fit(disp=False)
            
            # Make predictions for the next day
            forecast = self.arima_model.forecast(steps=1)
            last_price = self.stock_data['Close'].iloc[-1]
            pred_price = forecast[0]
            pred_return = (pred_price / last_price - 1) * 100
            
            print(f"📈 ARIMA forecast for next day: ${pred_price:.2f} ({pred_return:.2f}%)")
            
            return self.arima_model
        except Exception as e:
            print(f"⚠️ Error building ARIMA model: {str(e)}")
            return None
    
    def compare_models(self):
        """Compare all models and visualize results"""
        print("📊 Comparing model performance...")
        
        if not self.results:
            print("⚠️ No models to compare. Please train models first.")
            return pd.DataFrame()
    
        # Extract metrics for comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in model_names]
        precisions = [self.results[model]['precision'] for model in model_names]
        recalls = [self.results[model]['recall'] for model in model_names]
        f1_scores = [self.results[model]['f1_score'] for model in model_names]
        
        # Add profit metrics if available
        train_profits = []
        test_profits = []
        composite_scores = []
    
        for model in model_names:
            train_profits.append(self.results[model].get('train_profit', float('nan')))
            test_profits.append(self.results[model].get('test_profit', float('nan')))
            composite_scores.append(self.results[model].get('composite_score', float('nan')))
        
        # Create DataFrame for comparison
        comparison_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1_scores
        })
        
        # Add profit columns if available
        if not all(np.isnan(train_profits)):
            comparison_df['Training Profit'] = train_profits
        
        if not all(np.isnan(test_profits)):
            comparison_df['Test Profit'] = test_profits
            
        if not all(np.isnan(composite_scores)):
            comparison_df['Composite Score'] = composite_scores
            # Sort by composite score if available
            comparison_df = comparison_df.sort_values('Composite Score', ascending=False).reset_index(drop=True)
        else:
            # Otherwise sort by F1 score
            comparison_df = comparison_df.sort_values('F1 Score', ascending=False).reset_index(drop=True)
        
        print("\n🏆 Model Performance Ranking:")
        print(comparison_df)
    
        # Plot comparison
        plt.figure(figsize=(14, 8))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, comparison_df[metric], width, label=metric)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title(f'Model Comparison for {self.ticker} Stock Prediction')
        plt.xticks(x + width*1.5, comparison_df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot confusion matrices for top models
        plt.figure(figsize=(15, 10))
        
        top_models = comparison_df['Model'].tolist()[:min(3, len(model_names))]
        
        for i, model_name in enumerate(top_models):
            plt.subplot(1, len(top_models), i+1)
            
            # Handle different prediction keys for traditional vs deep learning models
            if model_name in ['LSTM', 'CNN-LSTM', 'Bidirectional GRU']:
                # For deep learning models
                if 'y_test_seq' in self.results[model_name]:
                    # Check whether we have 'predictions' or 'test_predictions'
                    if 'predictions' in self.results[model_name]:
                        pred_key = 'predictions'
                    elif 'test_predictions' in self.results[model_name]:
                        pred_key = 'test_predictions'
                    else:
                        # Skip if neither key exists
                        continue
                        
                    y_true = self.results[model_name]['y_test_seq']
                    y_pred = self.results[model_name][pred_key]
                    # Ensure lengths match - use minimum length
                    min_len = min(len(y_true), len(y_pred))
                    cm = confusion_matrix(y_true[:min_len], y_pred[:min_len])
                else:
                    # Skip if data is missing
                    continue
            else:
                # For traditional ML models, check which key exists
                if 'predictions' in self.results[model_name]:
                    pred_key = 'predictions'
                elif 'test_predictions' in self.results[model_name]:
                    pred_key = 'test_predictions'
                else:
                    # Skip if neither key exists
                    continue
                    
                # Ensure lengths match
                if len(self.y_test) == len(self.results[model_name][pred_key]):
                    cm = confusion_matrix(self.y_test, self.results[model_name][pred_key])
                else:
                    # Handle inconsistent lengths
                    min_len = min(len(self.y_test), len(self.results[model_name][pred_key]))
                    cm = confusion_matrix(
                        self.y_test.iloc[:min_len],
                        self.results[model_name][pred_key][:min_len]
                    )
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
        
        # If LSTM is in results, plot training history
        for model_name in ['LSTM', 'CNN-LSTM', 'Bidirectional GRU']:
            if model_name in self.results and 'history' in self.results[model_name]:
                plt.figure(figsize=(12, 5))
                history = self.results[model_name]['history']
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title(f'{model_name} Model Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'])
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title(f'{model_name} Model Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Validation'])
                
                plt.tight_layout()
                plt.show()
        
        return comparison_df


    

    def generate_trading_signals(self):
        """Generate trading signals using the best model"""
        print(f"💹 Generating trading signals using {self.best_model_name}...")
        
        if not self.best_model:
            print("⚠️ No best model selected. Please train models first.")
            return None
                
        # For traditional ML models
        if self.best_model_name not in ['LSTM', 'CNN-LSTM', 'Bidirectional GRU']:
            # Apply the model to all data points
            self.stock_data['ML_Signal'] = self.best_model.predict(self.X_scaled)
            self.stock_data['ML_Trade'] = self.stock_data['ML_Signal'].shift(1)
            
        else:
            # For deep learning models, we need a different approach since they use sequences
            # We'll use the trained model to predict on all possible sequences in our data
            
            # Get the sequence length from X_test_seq if available
            sequence_length = 60  # Default sequence length
            
            if self.best_model_name in self.results:
                # Try to get the sequence length from the model results
                if 'X_test_seq' in self.results[self.best_model_name]:
                    sequence_length = self.results[self.best_model_name]['X_test_seq'].shape[1]
            
            # Initialize signals array with NaNs
            signals = np.array([np.nan] * len(self.stock_data))
            
            # Prepare all data for sequence prediction
            all_sequences = []
            for i in range(sequence_length, len(self.X_scaled)):
                all_sequences.append(self.X_scaled[i-sequence_length:i])
            
            all_sequences = np.array(all_sequences)
            
            # Make predictions on all sequences
            dl_model = self.results[self.best_model_name]['model']
            all_proba = dl_model.predict(all_sequences)
            
            # Apply 3-way classification to all predictions
            for i, prob in enumerate(all_proba):
                idx = i + sequence_length  # Adjust index
                if idx < len(signals):
                    if prob > 0.65:  # Strong confidence for BUY
                        signals[idx] = 1    # BUY signal
                    elif prob < 0.35:  # Strong confidence for SELL
                        signals[idx] = 0    # SELL signal
                    else:
                        signals[idx] = 0.5  # HOLD signal
            
            # Add signals to data
            self.stock_data['ML_Signal'] = signals
            self.stock_data['ML_Trade'] = self.stock_data['ML_Signal'].shift(1)
        
        # Calculate potential returns from the strategy
        self.stock_data['Strategy_Return'] = self.stock_data['Returns'] * self.stock_data['ML_Trade']
        self.stock_data['Cumulative_Strategy_Return'] = (1 + self.stock_data['Strategy_Return']).cumprod()
        self.stock_data['Cumulative_Market_Return'] = (1 + self.stock_data['Returns']).cumprod()
        
        # Calculate actual portfolio values based on initial investment
        self.stock_data['Strategy_Value'] = self.initial_investment * self.stock_data['Cumulative_Strategy_Return'].fillna(1)
        self.stock_data['Market_Value'] = self.initial_investment * self.stock_data['Cumulative_Market_Return'].fillna(1)
        
        # Latest trading recommendation
        if len(self.stock_data) > 1:
            latest_trade = self.stock_data.iloc[-1]
#            prev_trade = self.stock_data.iloc[-2]
            
            print("\n🚨 TRADING RECOMMENDATION:")
            
            if not np.isnan(latest_trade['ML_Trade']):
                if latest_trade['ML_Trade'] == 1:
                    print("✅ BULLISH SIGNAL - Enter LONG position")
                    print(f"🚀 Entry Price: ${latest_trade['Close']:.2f}")
                    if 'Bollinger_Upper_20' in self.stock_data.columns:
                        print(f"🎯 Target Price: ${latest_trade['Bollinger_Upper_20']:.2f}")
                    if 'Bollinger_Lower_20' in self.stock_data.columns:
                        print(f"🛑 Stop-Loss Price: ${latest_trade['Bollinger_Lower_20']:.2f}")
                elif latest_trade['ML_Trade'] == 0.5:
                    print("🟡 HOLD SIGNAL - Maintain current positions")
                    print(f"💰 Current Price: ${latest_trade['Close']:.2f}")
                else:
                    print("⛔ BEARISH SIGNAL - Exit LONG position or Enter SHORT position")
                    print(f"🚀 Exit/Short Price: ${latest_trade['Close']:.2f}")
                    if 'Bollinger_Lower_20' in self.stock_data.columns:
                        print(f"🎯 Target Price: ${latest_trade['Bollinger_Lower_20']:.2f}")
                    if 'Bollinger_Upper_20' in self.stock_data.columns:
                        print(f"🛑 Stop-Loss Price: ${latest_trade['Bollinger_Upper_20']:.2f}")
            else:
                print("⚠️ No signal available for the latest data point")
        
        return self.stock_data

    
    def plot_trading_signals(self):
        """Visualize trading signals and strategy performance"""
        print("📈 Visualizing trading strategy...")
        
        if 'ML_Trade' not in self.stock_data.columns:
            print("⚠️ No trading signals generated. Please run generate_trading_signals() first.")
            return None
            
        # Plot price and trading signals
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Price and Trading Signals
        plt.subplot(2, 1, 1)
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', alpha=0.5)
        
        # Filter out NaN values for plotting
        buy_signals = self.stock_data[self.stock_data['ML_Trade'] == 1]
        sell_signals = self.stock_data[self.stock_data['ML_Trade'] == 0]
        
        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', 
                   label=f'{self.best_model_name} Buy Signal', alpha=1, s=100)
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', 
                   label=f'{self.best_model_name} Sell Signal', alpha=1, s=100)
        
        plt.title(f"{self.ticker} Trading Signals using {self.best_model_name}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Strategy Performance
        plt.subplot(2, 1, 2)
        plt.plot(self.stock_data.index, self.stock_data['Market_Value'], 
                label=f'Buy & Hold (${self.initial_investment:,})', color='blue')
        plt.plot(self.stock_data.index, self.stock_data['Strategy_Value'], 
                label=f'{self.best_model_name} Strategy (${self.initial_investment:,})', color='green')
        plt.title(f"{self.ticker} Strategy Performance Comparison")
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate strategy performance metrics
        final_market_value = self.stock_data['Market_Value'].iloc[-1]
        final_strategy_value = self.stock_data['Strategy_Value'].iloc[-1]
        
        final_bh_return = (final_market_value / self.initial_investment) - 1
        final_strategy_return = (final_strategy_value / self.initial_investment) - 1
        
        annual_factor = 252 / len(self.stock_data)
        bh_annual_return = (1 + final_bh_return) ** annual_factor - 1
        strategy_annual_return = (1 + final_strategy_return) ** annual_factor - 1
        
        bh_volatility = self.stock_data['Returns'].std() * np.sqrt(252)
        strategy_volatility = self.stock_data['Strategy_Return'].std() * np.sqrt(252)
        
        bh_sharpe = bh_annual_return / bh_volatility if bh_volatility > 0 else 0
        strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility > 0 else 0
        
        # Calculate max drawdown
        bh_cumulative = self.stock_data['Cumulative_Market_Return']
        strategy_cumulative = self.stock_data['Cumulative_Strategy_Return']
        
        bh_peak = bh_cumulative.expanding(min_periods=1).max()
        strategy_peak = strategy_cumulative.expanding(min_periods=1).max()
        
        bh_drawdown = (bh_cumulative / bh_peak - 1)
        strategy_drawdown = (strategy_cumulative / strategy_peak - 1)
        
        bh_max_drawdown = bh_drawdown.min()
        strategy_max_drawdown = strategy_drawdown.min()
        
        # Print performance summary
        print("\n📊 PERFORMANCE SUMMARY:")
        print(f"⏱️ Testing Period: {self.stock_data.index[0].strftime('%Y-%m-%d')} to {self.stock_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Initial Investment: ${self.initial_investment:,}")
        print(f"💵 Final Buy & Hold Value: ${final_market_value:,.2f}")
        print(f"💵 Final Strategy Value: ${final_strategy_value:,.2f}")
        print(f"📈 Buy & Hold Return: {final_bh_return:.2%}")
        print(f"📈 Strategy Return: {final_strategy_return:.2%}")
        print(f"💰 Strategy Profit/Loss: ${final_strategy_value - self.initial_investment:,.2f}")
        print(f"📊 Buy & Hold Annual Return: {bh_annual_return:.2%}")
        print(f"📊 Strategy Annual Return: {strategy_annual_return:.2%}")
        print(f"📉 Buy & Hold Volatility: {bh_volatility:.2%}")
        print(f"📉 Strategy Volatility: {strategy_volatility:.2%}")
        print(f"📉 Buy & Hold Max Drawdown: {bh_max_drawdown:.2%}")
        print(f"📉 Strategy Max Drawdown: {strategy_max_drawdown:.2%}")
        print(f"⚖️ Buy & Hold Sharpe Ratio: {bh_sharpe:.2f}")
        print(f"⚖️ Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        
        return {
            'Buy & Hold Return': final_bh_return,
            'Strategy Return': final_strategy_return,
            'Buy & Hold Annual Return': bh_annual_return,
            'Strategy Annual Return': strategy_annual_return,
            'Buy & Hold Sharpe': bh_sharpe,
            'Strategy Sharpe': strategy_sharpe,
            'Buy & Hold Max Drawdown': bh_max_drawdown,
            'Strategy Max Drawdown': strategy_max_drawdown,
            'Initial Investment': self.initial_investment,
            'Final Strategy Value': final_strategy_value,
            'Profit/Loss': final_strategy_value - self.initial_investment
        }
    
    def feature_importance(self):
        """Analyze feature importance for tree-based models"""
        if self.best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Random Forest_Tuned', 'Gradient Boosting_Tuned', 'XGBoost_Tuned']:
            print(f"🔍 Analyzing feature importance for {self.best_model_name}...")
            
            # Get feature importance
            model = self.results[self.best_model_name]['model']
            
            if 'XGBoost' in self.best_model_name:
                importance = model.feature_importances_
            else:
                importance = model.feature_importances_
                
            # Sort features by importance
            feature_names = self.X.columns
            indices = np.argsort(importance)[::-1]
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance for {self.best_model_name}')
            plt.bar(range(len(indices[:15])), importance[indices[:15]], align='center')
            plt.xticks(range(len(indices[:15])), [feature_names[i] for i in indices[:15]], rotation=90)
            plt.tight_layout()
            plt.show()
            
            # Print top 10 features
            print("\n🔝 TOP 10 IMPORTANT FEATURES:")
            for i, idx in enumerate(indices[:10]):
                print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
                
            return {feature_names[i]: importance[i] for i in indices}
        else:
            print(f"⚠️ Feature importance not available for {self.best_model_name}")
            return None


    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print("📊 Generating ROC curves for model comparison...")
        
        if not self.results:
            print("⚠️ No models to compare. Please train models first.")
            return None
            
        plt.figure(figsize=(12, 8))
        
        # Store AUC values for comparison
        auc_values = {}
        
        for model_name, results in self.results.items():
            if model_name in ['LSTM', 'CNN-LSTM', 'Bidirectional GRU']:
                # For deep learning models
                try:
                    model = results['model']
                    X_test_seq = results['X_test_seq']
                    y_test = results['y_test_seq']
                    
                    y_pred_proba = model.predict(X_test_seq)
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                    auc_values[model_name] = roc_auc
                except Exception as e:
                    print(f"⚠️ Error generating ROC curve for {model_name}: {str(e)}")
            else:
                # For traditional ML models
                try:
                    model = results['model']
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                        auc_values[model_name] = roc_auc
                except Exception as e:
                    print(f"⚠️ Error generating ROC curve for {model_name}: {str(e)}")
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
        
        # Print AUC summary
        print("\n📊 AUC Scores Summary:")
        for model_name, auc_value in sorted(auc_values.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name}: {auc_value:.4f}")
            
        return auc_values
        
    
    def evaluate_model_performance(self, model, X_data=None, y_data=None, is_sequence_model=False):
        """
        Central method to evaluate model performance consistently across all models.
        Can be used for both training and test evaluations.
        
        Parameters:
        -----------
        model : sklearn or keras model
            The trained model to evaluate
        X_data : array-like, optional
            Feature data to evaluate on. If None, uses the entire dataset.
        y_data : array-like, optional
            Target data for metrics calculation. If None, uses the entire dataset.
        is_sequence_model : bool
            Whether the model is a sequence-based model (LSTM, CNN-LSTM, GRU)
        
        Returns:
        --------
        dict : Dictionary containing performance metrics
        """
        print("Evaluating model performance...")
        
        # If no specific data provided, use the entire dataset
        if X_data is None:
            X_data = self.X_scaled
            y_data = self.y
        
        # Generate signals for the model
        temp_signals = np.zeros(len(self.stock_data))
        
        # Initialize proba as None to avoid reference errors
        proba = None
        
        if is_sequence_model:
            # For sequence models (LSTM, CNN-LSTM, BiGRU)
            sequence_length = 60  # Default sequence length
            
            # Prepare sequences for the evaluation data
            if hasattr(self, 'prepare_lstm_data'):
                # If using existing prepared data
                try:
                    # If prepare_lstm_data accepts sequence_length parameter
                    _, _, X_sequences, _, _ = self.prepare_lstm_data(sequence_length)
                except TypeError:
                    # If prepare_lstm_data doesn't accept sequence_length parameter
                    _, _, X_sequences, _, _ = self.prepare_lstm_data()
            else:
                # Create sequences for the dataset
                all_sequences = []
                for i in range(sequence_length, len(X_data)):
                    all_sequences.append(X_data[i-sequence_length:i])
                
                X_sequences = np.array(all_sequences)
                
            # Make predictions
            proba = model.predict(X_sequences)
            
            # Apply 3-way classification
            for i, prob in enumerate(proba):
                idx = i + sequence_length  # Adjust index
                if idx < len(temp_signals):
                    if prob > 0.65:  # Strong confidence for BUY
                        temp_signals[idx] = 1    # BUY signal
                    elif prob < 0.35:  # Strong confidence for SELL
                        temp_signals[idx] = 0    # SELL signal
                    else:
                        temp_signals[idx] = 0.5  # HOLD signal
        else:
            # For traditional ML models
            if hasattr(model, 'predict_proba') and not isinstance(model, SVC):
                # For models that provide probability estimates
                proba = model.predict_proba(X_data)
                
                for i, prob in enumerate(proba[:, 1]):  # Class 1 probability
                    if i < len(temp_signals):
                        if prob > 0.65:  # Strong confidence for BUY
                            temp_signals[i] = 1
                        elif prob < 0.35:  # Strong confidence for SELL
                            temp_signals[i] = 0
                        else:
                            temp_signals[i] = 0.5
            else:
                # For models without probability estimates
                preds = model.predict(X_data)
                proba = preds  # Store predictions as proba for metrics calculation
                for i, pred in enumerate(preds):
                    if i < len(temp_signals):
                        temp_signals[i] = pred
        
        # Create a temporary dataframe to calculate returns
        temp_df = self.stock_data.copy()
        temp_df['ML_Signal'] = temp_signals
        temp_df['ML_Trade'] = temp_df['ML_Signal'].shift(1)
        
        # Calculate returns
        temp_df['Strategy_Return'] = temp_df['Returns'] * temp_df['ML_Trade']
        temp_df['Cumulative_Strategy_Return'] = (1 + temp_df['Strategy_Return']).cumprod()
        temp_df['Cumulative_Market_Return'] = (1 + temp_df['Returns']).cumprod()
        
        # Calculate portfolio values
        temp_df['Strategy_Value'] = self.initial_investment * temp_df['Cumulative_Strategy_Return'].fillna(1)
        temp_df['Market_Value'] = self.initial_investment * temp_df['Cumulative_Market_Return'].fillna(1)
        
        # Define training and test periods
        test_size = len(self.X_test)
        train_end_idx = len(temp_df) - test_size
        
        # Get test data
        if train_end_idx > 0:
            test_data = temp_df.iloc[-test_size:]
        else:
            test_data = temp_df  # Use all data if train_end_idx is not positive
        
        # Get train data (exclude test data)
        if train_end_idx > 0:
            train_data = temp_df.iloc[:train_end_idx]
        else:
            train_data = pd.DataFrame()  # Empty DataFrame if no training data
        
        # Calculate classification metrics if y_data is provided
        metrics = {}
        if y_data is not None and proba is not None:
            if is_sequence_model:
                # For sequence models, adjust indices
                if hasattr(self, 'y_test') and len(self.y_test) == len(proba):
                    y_true = self.y_test
                else:
                    # Use available data, but could be misaligned
                    y_true = y_data[-len(proba):] if len(proba) < len(y_data) else y_data
            else:
                y_true = y_data
            
            # Binary predictions for metrics
            if hasattr(proba, 'ndim') and proba.ndim > 1:
                # For probability arrays
                y_pred_binary = (proba[:, 1] > 0.5).astype(int) if proba.shape[1] > 1 else (proba > 0.5).astype(int).flatten()
            else:
                # For flat arrays
                y_pred_binary = (proba > 0.5).astype(int)
            
            # Ensure lengths match
            min_len = min(len(y_true), len(y_pred_binary))
            y_true = y_true[:min_len]
            y_pred_binary = y_pred_binary[:min_len]
            
            # Calculate metrics
            try:
                metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
                metrics['precision'] = precision_score(y_true, y_pred_binary)
                metrics['recall'] = recall_score(y_true, y_pred_binary)
                metrics['f1_score'] = f1_score(y_true, y_pred_binary)
            except Exception as e:
                print(f"⚠️ Error calculating classification metrics: {str(e)}")
                # Set default values
                metrics['accuracy'] = 0
                metrics['precision'] = 0
                metrics['recall'] = 0
                metrics['f1_score'] = 0
        
        # Calculate training period performance
        if len(train_data) > 0:
            train_start_val = train_data['Strategy_Value'].iloc[0]
            train_end_val = train_data['Strategy_Value'].iloc[-1]
            train_profit = train_end_val - train_start_val
            train_return = (train_end_val / train_start_val) - 1
        else:
            train_profit = 0
            train_return = 0
        
        # Calculate test period performance
        if len(test_data) > 0:
            # Start from test period beginning
            test_start_val = test_data['Strategy_Value'].iloc[0]
            test_end_val = test_data['Strategy_Value'].iloc[-1]
            test_profit = test_end_val - test_start_val
            test_return = (test_end_val / test_start_val) - 1
            
            # Market performance
            market_start_val = test_data['Market_Value'].iloc[0]
            market_end_val = test_data['Market_Value'].iloc[-1]
            market_return = (market_end_val / market_start_val) - 1
            
            # Alpha (excess return)
            alpha = test_return - market_return
            
            # Calculate volatility
            test_strategy_vol = test_data['Strategy_Return'].std() * np.sqrt(252)
            test_market_vol = test_data['Returns'].std() * np.sqrt(252)
            
            # Sharpe ratio
            test_strategy_sharpe = (test_return / len(test_data) * 252) / test_strategy_vol if test_strategy_vol > 0 else 0
            test_market_sharpe = (market_return / len(test_data) * 252) / test_market_vol if test_market_vol > 0 else 0
            
            # Win rate
            win_rate = (test_data['Strategy_Return'] > 0).sum() / len(test_data[~test_data['Strategy_Return'].isna()])
            
            # Profit factor
            positive_returns = test_data['Strategy_Return'][test_data['Strategy_Return'] > 0].sum()
            negative_returns = abs(test_data['Strategy_Return'][test_data['Strategy_Return'] < 0].sum())
            profit_factor = positive_returns / negative_returns if negative_returns > 0 else float('inf')
            
            # Annualize returns
            days = len(test_data)
            annual_factor = 252 / days
            test_annual_return = (1 + test_return) ** annual_factor - 1
            market_annual_return = (1 + market_return) ** annual_factor - 1
        else:
            test_profit = 0
            test_return = 0
            market_return = 0
            alpha = 0
            test_strategy_vol = 0
            test_market_vol = 0
            test_strategy_sharpe = 0
            test_market_sharpe = 0
            win_rate = 0
            profit_factor = 0
            days = 0
            test_annual_return = 0
            market_annual_return = 0
        
        # Combine metrics
        metrics.update({
            'train_profit': train_profit,
            'train_return': train_return,
            'test_profit': test_profit,
            'test_return': test_return,
            'market_return': market_return,
            'alpha': alpha,
            'strategy_volatility': test_strategy_vol,
            'market_volatility': test_market_vol,
            'strategy_sharpe': test_strategy_sharpe,
            'market_sharpe': test_market_sharpe,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'test_period_days': days,
            'strategy_annual_return': test_annual_return,
            'market_annual_return': market_annual_return
        })
        
        # Calculate composite score
        if 'f1_score' in metrics and 'train_return' in metrics and 'test_return' in metrics:
            f1 = metrics['f1_score']
            train_return = metrics['train_return']
            test_return = metrics['test_return']
            
            # Simple composite score with weights
            metrics['composite_score'] = (
                0.3 * f1 +             # F1 score weight
                0.3 * train_return +   # Training return weight
                0.4 * test_return      # Test return weight (higher weight for out-of-sample performance)
            )
        
        # Debug info
        print(f"Train Return: {train_return:.2%}, Test Return: {test_return:.2%}")
        
        return metrics, temp_df

    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        self.download_data()
        self.calculate_technical_indicators()
        self.prepare_features()
        self.build_models()    
        self.train_evaluate_models()

        # Add LSTM if enabled
        if self.enabled_models == "all" or self.enabled_models == "deep_learning" or "LSTM" in self.enabled_models:
            self.train_lstm_model()
        
        self.compare_models()
        self.generate_trading_signals()
        self.plot_trading_signals()
        
        # Feature importance for tree-based models
        if self.best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Random Forest_Tuned', 'Gradient Boosting_Tuned', 'XGBoost_Tuned']:
            self.feature_importance()
        
        return self.results, self.best_model_name
    
    def run_advanced_analysis(self):
        """Run advanced analysis with all enhanced models"""
        print("🚀 Running advanced stock analysis...")
        
        # Run the basic pipeline first
        self.download_data()
        self.calculate_technical_indicators()
        self.prepare_features()
        self.build_models()
        # Train base models
        self.train_evaluate_models()
                
        
        # Train deep learning models if enabled
        if self.enabled_models == "all" or self.enabled_models == "deep_learning" or any(m in self.enabled_models for m in ["LSTM", "CNN-LSTM", "Bidirectional GRU"]):
            print("\n🧠 Training deep learning models...")
            
            if self.enabled_models == "all" or self.enabled_models == "deep_learning" or "LSTM" in self.enabled_models:
                try:
                    self.train_lstm_model(units=64, epochs=100)
                except Exception as e:
                    print(f"⚠️ Error training LSTM model: {str(e)}")
                
            if self.enabled_models == "all" or self.enabled_models == "deep_learning" or "CNN-LSTM" in self.enabled_models:
                try:
                    self.build_cnn_lstm_model()
                except Exception as e:
                    print(f"⚠️ Error building CNN-LSTM model: {str(e)}")
                
            if self.enabled_models == "all" or self.enabled_models == "deep_learning" or "Bidirectional GRU" in self.enabled_models:
                try:
                    self.build_bidirectional_gru_model()
                except Exception as e:
                    print(f"⚠️ Error building Bidirectional GRU model: {str(e)}")
        
        # Perform hyperparameter tuning on best traditional model
        best_trad_model = None
        best_trad_score = 0
        for model_name, results in self.results.items():
            if model_name not in ['LSTM', 'CNN-LSTM', 'Bidirectional GRU']:
                if 'composite_score' in results:
                    model_score = results['composite_score']
                else:
                    model_score = results['f1_score']
                    
                if model_score > best_trad_score:
                    best_trad_model = model_name
                    best_trad_score = model_score
        
        if best_trad_model and (self.enabled_models == "all" or self.enabled_models == "traditional" or best_trad_model in self.enabled_models):
            try:
                print(f"\n🔧 Tuning best traditional model: {best_trad_model}...")
                self.perform_hyperparameter_tuning(best_trad_model)
            except Exception as e:
                print(f"⚠️ Error during hyperparameter tuning: {str(e)}")
        
        # Build ARIMA model for time series forecasting if available
        if ARIMA_AVAILABLE:
            try:
                self.build_arima_model()
            except Exception as e:
                print(f"⚠️ ARIMA model building failed: {str(e)}")
            
        # Compare all models
        comparison_df = self.compare_models()
        
        # Generate trading signals
        self.generate_trading_signals()
        performance = self.plot_trading_signals()
        
        # Estimate returns for test period using the best model
        test_performance = self.estimate_test_returns()
        
        # Analyze feature importance for tree-based models
        if self.best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Random Forest_Tuned', 'Gradient Boosting_Tuned', 'XGBoost_Tuned']:
            self.feature_importance()
        
        # Perform ROC curve analysis
        try:
            self.plot_roc_curves()
        except Exception as e:
            print(f"⚠️ Error generating ROC curves: {str(e)}")
            
        # Check what comparison_df and performance contain
        if not isinstance(comparison_df, pd.DataFrame):
            comparison_df = pd.DataFrame()
        
        if not isinstance(performance, dict):
            # Create default performance dict if not available
            performance = {
                'Strategy Return': 0,
                'Buy & Hold Return': 0,
                'Strategy Annual Return': 0,
                'Buy & Hold Annual Return': 0,
                'Strategy Sharpe': 0,
                'Buy & Hold Sharpe': 0,
                'Strategy Max Drawdown': 0,
                'Buy & Hold Max Drawdown': 0,
                'Initial Investment': self.initial_investment,
                'Final Strategy Value': self.initial_investment,
                'Profit/Loss': 0
            }            

        return comparison_df, performance, test_performance, self.best_model_name

 
    def estimate_test_returns(self):
        """
        Estimate and analyze returns for the test period using the best model
        Returns detailed metrics on model performance during test period
        """
        print("📊 Estimating test period returns using best model...")
        
        if not self.best_model:
            print("⚠️ No best model selected. Please train models first.")
            return None
            
        # Get the test period data
        test_data = self.stock_data.iloc[-len(self.y_test):]
        
        # Make sure we have ML signals for the test period
        if 'ML_Signal' not in test_data.columns:
            print("⚠️ Trading signals not generated. Generating signals first...")
            self.generate_trading_signals()
            test_data = self.stock_data.iloc[-len(self.y_test):]
        
        # Calculate returns metrics specifically for test period
        test_data['Market_Return'] = test_data['Returns']
        test_data['Strategy_Return'] = test_data['Returns'] * test_data['ML_Trade']
        
        # Calculate cumulative returns
        test_data['Cumulative_Market_Return'] = (1 + test_data['Market_Return']).cumprod()
        test_data['Cumulative_Strategy_Return'] = (1 + test_data['Strategy_Return']).cumprod()
        
        # Calculate portfolio values
        test_data['Market_Value'] = self.initial_investment * test_data['Cumulative_Market_Return']
        test_data['Strategy_Value'] = self.initial_investment * test_data['Cumulative_Strategy_Return']
        
        # Get final values
        initial_value = self.initial_investment
        final_market_value = test_data['Market_Value'].iloc[-1]
        final_strategy_value = test_data['Strategy_Value'].iloc[-1]
        
        # Calculate returns
        market_return = (final_market_value / initial_value) - 1
        strategy_return = (final_strategy_value / initial_value) - 1
        
        # Calculate annualized returns
        days_in_test = len(test_data)
        annual_factor = 252 / days_in_test
        market_annual_return = (1 + market_return) ** annual_factor - 1
        strategy_annual_return = (1 + strategy_return) ** annual_factor - 1
        
        # Calculate volatility
        market_volatility = test_data['Market_Return'].std() * np.sqrt(252)
        strategy_volatility = test_data['Strategy_Return'].std() * np.sqrt(252)
        
        # Calculate Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
        market_sharpe = market_annual_return / market_volatility if market_volatility > 0 else 0
        strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility > 0 else 0
        
        # Calculate max drawdown
        market_peak = test_data['Cumulative_Market_Return'].expanding(min_periods=1).max()
        strategy_peak = test_data['Cumulative_Strategy_Return'].expanding(min_periods=1).max()
        
        market_drawdown = (test_data['Cumulative_Market_Return'] / market_peak - 1)
        strategy_drawdown = (test_data['Cumulative_Strategy_Return'] / strategy_peak - 1)
        
        market_max_drawdown = market_drawdown.min()
        strategy_max_drawdown = strategy_drawdown.min()
        
        # Calculate win rate
        winning_trades = test_data[test_data['Strategy_Return'] > 0]
        win_rate = len(winning_trades) / len(test_data) if len(test_data) > 0 else 0
        
        # Calculate profit factor
        gross_profit = test_data.loc[test_data['Strategy_Return'] > 0, 'Strategy_Return'].sum()
        gross_loss = abs(test_data.loc[test_data['Strategy_Return'] < 0, 'Strategy_Return'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Calculate average profit/loss per trade
        avg_profit = test_data['Strategy_Return'].mean()

        final_bh_return = (final_market_value / self.initial_investment) - 1
        final_strategy_return = (final_strategy_value / self.initial_investment) - 1
        
        annual_factor = 252 / len(self.stock_data)
        bh_annual_return = (1 + final_bh_return) ** annual_factor - 1
        strategy_annual_return = (1 + final_strategy_return) ** annual_factor - 1
        
        bh_volatility = self.stock_data['Returns'].std() * np.sqrt(252)
        strategy_volatility = self.stock_data['Strategy_Return'].std() * np.sqrt(252)
        
        bh_sharpe = bh_annual_return / bh_volatility if bh_volatility > 0 else 0
        strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility > 0 else 0
        
        # Calculate max drawdown
        bh_cumulative = self.stock_data['Cumulative_Market_Return']
        strategy_cumulative = self.stock_data['Cumulative_Strategy_Return']
        
        bh_peak = bh_cumulative.expanding(min_periods=1).max()
        strategy_peak = strategy_cumulative.expanding(min_periods=1).max()
        
        bh_drawdown = (bh_cumulative / bh_peak - 1)
        strategy_drawdown = (strategy_cumulative / strategy_peak - 1)
        
        bh_max_drawdown = bh_drawdown.min()
        strategy_max_drawdown = strategy_drawdown.min()
        
        # Plot test period performance
        plt.figure(figsize=(16, 10))
        
        # Plot 1: Test Period Trading Signals
        plt.subplot(2, 1, 1)
        plt.plot(test_data.index, test_data['Close'], label='Close Price', alpha=0.7)
        
        # Filter out NaN values for plotting
        buy_signals = test_data[test_data['ML_Trade'] == 1]
        sell_signals = test_data[test_data['ML_Trade'] == 0]
        
        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', 
                    label=f'{self.best_model_name} Buy Signal', alpha=1, s=100)
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', 
                    label=f'{self.best_model_name} Sell Signal', alpha=1, s=100)
        
        plt.title(f"{self.ticker} Test Period Trading Signals using {self.best_model_name}")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Test Period Performance
        plt.subplot(2, 1, 2)
        plt.plot(test_data.index, test_data['Market_Value'], 
                label=f'Buy & Hold (${initial_value:,})', color='blue')
        plt.plot(test_data.index, test_data['Strategy_Value'], 
                label=f'{self.best_model_name} Strategy (${initial_value:,})', color='green')
        plt.title(f"{self.ticker} Test Period Performance")
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot monthly returns comparison
        if len(test_data) > 20:  # Only if we have enough data
            # Resample to monthly returns
            test_data.index = pd.to_datetime(test_data.index)
            market_monthly = test_data['Market_Return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            strategy_monthly = test_data['Strategy_Return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            # Plot monthly returns
            plt.figure(figsize=(14, 6))
            
            months = market_monthly.index.strftime('%Y-%m')
            x = np.arange(len(months))
            width = 0.35
            
            plt.bar(x - width/2, market_monthly.values * 100, width, label='Buy & Hold', color='blue', alpha=0.7)
            plt.bar(x + width/2, strategy_monthly.values * 100, width, label=f'{self.best_model_name} Strategy', color='green', alpha=0.7)
            
            plt.xlabel('Month')
            plt.ylabel('Monthly Return (%)')
            plt.title(f'{self.ticker} Monthly Returns Comparison (Test Period)')
            plt.xticks(x, months, rotation=45)
            plt.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.show()
        
        # Print performance summary
        print("\n📊 TEST PERIOD PERFORMANCE SUMMARY:")
        print(f"⏱️ Test Period: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ({days_in_test} trading days)")
        print(f"💰 Initial Investment: ${initial_value:,}")
        print(f"💵 Final Buy & Hold Value: ${final_market_value:,.2f}")
        print(f"💵 Final Strategy Value: ${final_strategy_value:,.2f}")
        print(f"📈 Buy & Hold Return: {market_return:.2%}")
        print(f"📈 Strategy Return: {strategy_return:.2%}")
        print(f"📊 Buy & Hold Annual Return: {market_annual_return:.2%}")
        print(f"📊 Strategy Annual Return: {strategy_annual_return:.2%}")
        print(f"📉 Buy & Hold Volatility: {market_volatility:.2%}")
        print(f"📉 Strategy Volatility: {strategy_volatility:.2%}")
        print(f"📉 Buy & Hold Max Drawdown: {market_max_drawdown:.2%}")
        print(f"📉 Strategy Max Drawdown: {strategy_max_drawdown:.2%}")
        print(f"⚖️ Buy & Hold Sharpe Ratio: {market_sharpe:.2f}")
        print(f"⚖️ Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"🎯 Win Rate: {win_rate:.2%}")
        print(f"💰 Profit Factor: {profit_factor:.2f}")
        print(f"💵 Average Return per Trade: {avg_profit:.2%}")
        
        # Calculate alpha and beta
        benchmark_returns = test_data['Market_Return']
        strategy_returns = test_data['Strategy_Return']
        
        # Calculate covariance and beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Calculate alpha (Jensen's Alpha)
        risk_free_rate = 0  # Assuming 0 for simplicity
        alpha = strategy_annual_return - (risk_free_rate + beta * (market_annual_return - risk_free_rate))
        
        print(f"📊 Strategy Alpha: {alpha:.2%}")
        print(f"📊 Strategy Beta: {beta:.2f}")
        
        # Return a dictionary with all performance metrics
        return {
            'Test Period': f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}",
            'Trading Days': days_in_test,
            'Initial Investment': initial_value,
            'Final Market Value': final_market_value,
            'Final Strategy Value': final_strategy_value,
            'Market Return': market_return,
            'Strategy Return': strategy_return,
            'Market Annual Return': market_annual_return,
            'Strategy Annual Return': strategy_annual_return,
            'Market Volatility': market_volatility,
            'Strategy Volatility': strategy_volatility,
            'Market Max Drawdown': market_max_drawdown,
            'Strategy Max Drawdown': strategy_max_drawdown,
            'Market Sharpe': market_sharpe,
            'Strategy Sharpe': strategy_sharpe,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Average Return per Trade': avg_profit,
            'Alpha': alpha,
            'Beta': beta,
            'Test Data': test_data,             
            'Test Period Days': days_in_test ,             
            'Buy & Hold Return': final_bh_return,
            'Buy & Hold Annual Return': bh_annual_return,
            'Buy & Hold Sharpe': bh_sharpe,
            'Buy & Hold Max Drawdown': bh_max_drawdown,
            'Profit/Loss': final_strategy_value - self.initial_investment
        }




# Execute the analysis with user inputs
if __name__ == "__main__":
    print("=" * 80)
    print("🔍 ADVANCED STOCK ANALYSIS")
    print("=" * 80)
    
    # Get user inputs
    ticker = input("Enter stock ticker (default: IOVA): ") or "IOVA"
    
    default_start = "2018-01-01"
    default_end = datetime.now().strftime("%Y-%m-%d")
    start_date = input(f"Enter start date (YYYY-MM-DD) (default: {default_start}): ") or default_start
    end_date = input(f"Enter end date (YYYY-MM-DD) (default: {default_end}): ") or default_end
    
    try:
        test_period = int(input("Enter test period in months (default: 6): ") or "6")
    except ValueError:
        print("Invalid input, using default 6 months")
        test_period = 6
    
    try:
        initial_investment = float(input("Enter initial investment amount in $ (default: 10000): ") or "10000")
    except ValueError:
        print("Invalid input, using default $10,000")
        initial_investment = 10000
    
    # Get model preferences
    print("\nModel selection:")
    print("1. All models (default)")
    print("2. Traditional ML models only (faster)")
    print("3. Deep learning models only")
    print("4. Custom model selection")
    model_choice = input("Enter choice (1-4): ") or "1"
    
    # Get technical indicator preferences
    print("\nTechnical indicators:")
    print("1. All indicators (default)")
    print("2. Basic indicators only (faster)")
    indicator_choice = input("Enter choice (1-2): ") or "1"
    
    # Get prediction target preferences
    print("\nPrediction target:")
    print("1. Next day price direction (default)")
    print("2. Next day return > threshold")
    print("3. 3-day return direction")
    print("4. 5-day return direction")
    target_choice = input("Enter choice (1-4): ") or "1"
    
    # Create instance with user inputs
    model_comparison = StockModelComparison(
        ticker=ticker, 
        start_date=start_date,
        end_date=end_date,
        test_period_months=test_period,
        initial_investment=initial_investment
    )
    
    # Configure based on user preferences
    model_comparison.set_enabled_models(model_choice)
    model_comparison.set_indicator_level(indicator_choice)
    model_comparison.set_target_type(target_choice)
    
    # Print configuration summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS CONFIGURATION:")
    print(f"🔍 Stock: {ticker}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"🧪 Test Period: {test_period} months")
    print(f"💰 Initial Investment: ${initial_investment:,.2f}")
    
    models_str = "All models" if model_comparison.enabled_models == "all" else \
                 "Traditional models only" if model_comparison.enabled_models == "traditional" else \
                 "Deep learning models only" if model_comparison.enabled_models == "deep_learning" else \
                 f"Custom selection: {', '.join(model_comparison.enabled_models)}"
    print(f"🤖 Models: {models_str}")
    
    indicators_str = "All indicators" if model_comparison.indicator_level == "all" else "Basic indicators only"
    print(f"📈 Indicators: {indicators_str}")
    
    target_str = "Next day direction" if model_comparison.target_type == "next_day_direction" else \
                f"Return > {model_comparison.target_params['threshold']*100}%" if model_comparison.target_type == "threshold" else \
                f"{model_comparison.target_params['days']}-day direction"
    print(f"🎯 Target: {target_str}")
    print("=" * 80)
    
    # Run analysis
    print("\n🚀 Starting analysis...")
    comparison_df, overall_performance, test_performance, best_model = model_comparison.run_advanced_analysis()
    
    # Display results
    print("\n" + "=" * 80)
    print(f"✨ FINAL RESULT: The best model for {ticker} is {best_model} ✨")
    print("\n📊 OVERALL PERFORMANCE:")
    print(f"💰 Initial Investment: ${overall_performance['Initial Investment']:,.2f}")
    print(f"💵 Final Value: ${overall_performance['Final Strategy Value']:,.2f}")
    print(f"📈 Total Return: {overall_performance['Strategy Return']:.2%}")
    print(f"💰 Profit/Loss: ${overall_performance['Profit/Loss']:,.2f}")
    print(f"📊 Sharpe Ratio: {overall_performance['Strategy Sharpe']:.2f}")
    print(f"📉 Max Drawdown: {overall_performance['Strategy Max Drawdown']:.2%}")

    print("\n📊 TEST PERIOD PERFORMANCE:")
    print(f"⏱️ Period: {test_performance['Test Period']}")
    print(f"📈 Strategy Return: {test_performance['Strategy Return']:.2%}")
    print(f"📈 Market Return: {test_performance['Market Return']:.2%}")
    print(f"📊 Alpha: {test_performance['Alpha']:.2%}")
    print(f"📊 Beta: {test_performance['Beta']:.2f}")
    print(f"🎯 Win Rate: {test_performance['Win Rate']:.2%}")
    print(f"💰 Profit Factor: {test_performance['Profit Factor']:.2f}")
    print("=" * 80)
