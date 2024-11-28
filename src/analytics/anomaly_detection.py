
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeFiAnomalyDetector:
    def __init__(self):
        """Initialize the anomaly detector."""
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.logger = logger
        self.feature_columns = [
            'value',
            'gas_price',
            'gas_used',
            'timestamp_hour',
            'timestamp_day',
            'volume_1h',
            'price_change_1h'
        ]

    def prepare_features(self, transactions: List[Dict]) -> Optional[pd.DataFrame]:
        """Prepare features for anomaly detection."""
        try:
            # Convert transactions to DataFrame
            df = pd.DataFrame(transactions)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Extract time-based features
            df['timestamp_hour'] = df['timestamp'].dt.hour
            df['timestamp_day'] = df['timestamp'].dt.dayofweek
            
            # Calculate rolling statistics
            df['volume_1h'] = df['value'].rolling(window='1H').sum()
            df['price_change_1h'] = df['value'].pct_change(periods=6)  # Assuming 10-minute intervals
            
            # Fill missing values
            df = df.fillna(method='ffill')
            
            # Select and scale features
            features = df[self.feature_columns]
            scaled_features = self.scaler.fit_transform(features)
            
            return pd.DataFrame(scaled_features, columns=self.feature_columns)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return None

    def train_model(self, transactions: List[Dict]):
        """Train the anomaly detection model."""
        try:
            features = self.prepare_features(transactions)
            if features is not None:
                self.isolation_forest.fit(features)
                self.logger.info("Successfully trained anomaly detection model")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")

    def detect_anomalies(self, transactions: List[Dict]) -> List[Dict]:
        """Detect anomalies in transactions."""
        try:
            features = self.prepare_features(transactions)
            if features is None:
                return []
            
            # Predict anomalies
            predictions = self.isolation_forest.predict(features)
            
            # Get anomaly scores
            scores = self.isolation_forest.score_samples(features)
            
            # Filter anomalous transactions
            anomalies = []
            df = pd.DataFrame(transactions)
            
            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1:  # Anomaly
                    anomalies.append({
                        **transactions[i],
                        'anomaly_score': float(score),
                        'anomaly_type': self._classify_anomaly(transactions[i], score)
                    })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {str(e)}")
            return []

    def _classify_anomaly(self, transaction: Dict, score: float) -> str:
        """Classify the type of anomaly."""
        try:
            # Define thresholds for different types of anomalies
            value_threshold = 1000  # ETH
            gas_threshold = 1000000  # gas units
            rapid_tx_threshold = 10  # transactions per minute
            
            anomaly_types = []
            
            # Check for high value transaction
            if float(transaction.get('value', 0)) > value_threshold:
                anomaly_types.append('high_value_transaction')
            
            # Check for high gas usage
            if float(transaction.get('gas_used', 0)) > gas_threshold:
                anomaly_types.append('high_gas_usage')
            
            # Check for flash loan patterns
            if self._is_flash_loan_pattern(transaction):
                anomaly_types.append('potential_flash_loan')
            
            # Check for reentrancy patterns
            if self._is_reentrancy_pattern(transaction):
                anomaly_types.append('potential_reentrancy')
            
            # If no specific type is identified, classify based on score
            if not anomaly_types:
                if score < -0.8:
                    anomaly_types.append('severe_anomaly')
                else:
                    anomaly_types.append('moderate_anomaly')
            
            return ', '.join(anomaly_types)
            
        except Exception as e:
            self.logger.error(f"Error classifying anomaly: {str(e)}")
            return 'unknown'

    def _is_flash_loan_pattern(self, transaction: Dict) -> bool:
        """Detect potential flash loan patterns."""
        try:
            # Check for typical flash loan characteristics
            # 1. Large value transferred
            # 2. Multiple contract interactions in single transaction
            # 3. Value returned to original sender
            
            if 'input_data' not in transaction:
                return False
            
            input_data = transaction['input_data']
            
            # Check for common flash loan signatures
            flash_loan_signatures = [
                'flashLoan(',
                'executeOperation(',
                'FLASHLOAN_PREMIUM_TOTAL'
            ]
            
            return any(sig in input_data for sig in flash_loan_signatures)
            
        except Exception as e:
            self.logger.error(f"Error checking flash loan pattern: {str(e)}")
            return False

    def _is_reentrancy_pattern(self, transaction: Dict) -> bool:
        """Detect potential reentrancy patterns."""
        try:
            # Check for typical reentrancy characteristics
            # 1. Multiple calls to same contract
            # 2. State changes after external calls
            # 3. Recursive patterns in trace
            
            if 'trace' not in transaction:
                return False
            
            trace = transaction['trace']
            
            # Check for repeated calls to same address
            addresses = [call['to'] for call in trace]
            address_counts = pd.Series(addresses).value_counts()
            
            return any(count > 3 for count in address_counts)
            
        except Exception as e:
            self.logger.error(f"Error checking reentrancy pattern: {str(e)}")
            return False

    def get_anomaly_statistics(self, timeframe: str = '24h') -> Dict:
        """Get statistics about detected anomalies."""
        try:
            now = datetime.now()
            if timeframe == '24h':
                start_time = now - timedelta(days=1)
            elif timeframe == '7d':
                start_time = now - timedelta(days=7)
            else:
                start_time = now - timedelta(hours=1)
            
            # Query recent anomalies
            recent_anomalies = [a for a in self.detected_anomalies 
                              if a['timestamp'] >= start_time.timestamp()]
            
            return {
                'total_anomalies': len(recent_anomalies),
                'anomaly_types': pd.Series([a['anomaly_type'] for a in recent_anomalies]).value_counts().to_dict(),
                'average_anomaly_score': np.mean([a['anomaly_score'] for a in recent_anomalies]),
                'high_risk_anomalies': len([a for a in recent_anomalies if a['anomaly_score'] < -0.8]),
                'timeframe': timeframe
            }
            
        except Exception as e:
            self.logger.error(f"Error getting anomaly statistics: {str(e)}")
            return {}