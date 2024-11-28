
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeFiOptimizer:
    def __init__(self):
        """Initialize DeFi optimizer."""
        self.logger = logger
        self.gas_model = LinearRegression()
        self.historical_data = []
        self.optimization_results = {}

    def analyze_gas_efficiency(self, transactions: List[Dict]) -> Dict:
        """Analyze gas efficiency of transactions."""
        try:
            df = pd.DataFrame(transactions)
            
            # Calculate gas efficiency metrics
            gas_metrics = {
                'average_gas_used': df['gas_used'].mean(),
                'median_gas_used': df['gas_used'].median(),
                'gas_used_std': df['gas_used'].std(),
                'total_gas_cost': (df['gas_used'] * df['gas_price']).sum(),
                'gas_price_correlation': df['gas_used'].corr(df['gas_price'])
            }
            
            # Identify optimal gas price windows
            gas_metrics['optimal_gas_windows'] = self._find_optimal_gas_windows(df)
            
            return gas_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing gas efficiency: {str(e)}")
            return {}

    def _find_optimal_gas_windows(self, df: pd.DataFrame) -> List[Dict]:
        """Find optimal time windows for gas prices."""
        try:
            df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
            hourly_stats = df.groupby('hour').agg({
                'gas_price': ['mean', 'std'],
                'gas_used': ['mean', 'count']
            }).reset_index()
            
            # Find windows with low gas prices and high success rates
            optimal_windows = []
            for _, row in hourly_stats.iterrows():
                if (row[('gas_price', 'mean')] < hourly_stats[('gas_price', 'mean')].mean() and 
                    row[('gas_used', 'count')] > hourly_stats[('gas_used', 'count')].mean()):
                    optimal_windows.append({
                        'hour': row['hour'],
                        'average_gas_price': float(row[('gas_price', 'mean')]),
                        'transaction_count': int(row[('gas_used', 'count')]),
                        'confidence_score': self._calculate_window_confidence(row)
                    })
            
            return optimal_windows
            
        except Exception as e:
            self.logger.error(f"Error finding optimal gas windows: {str(e)}")
            return []

    def _calculate_window_confidence(self, stats: pd.Series) -> float:
        """Calculate confidence score for a time window."""
        try:
            # Normalize metrics
            gas_price_score = 1 - (stats[('gas_price', 'mean')] / stats[('gas_price', 'std')])
            transaction_score = stats[('gas_used', 'count')] / stats[('gas_used', 'mean')]
            
            # Combine scores with weights
            confidence = (0.7 * gas_price_score + 0.3 * transaction_score)
            
            return float(np.clip(confidence, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Error calculating window confidence: {str(e)}")
            return 0.0

    def optimize_cross_layer_routing(self, eth_transactions: List[Dict], 
                                   stark_transactions: List[Dict]) -> Dict:
        """Optimize transaction routing between layers."""
        try:
            eth_df = pd.DataFrame(eth_transactions)
            stark_df = pd.DataFrame(stark_transactions)
            
            # Calculate layer-specific metrics
            eth_metrics = self._calculate_layer_metrics(eth_df, 'ethereum')
            stark_metrics = self._calculate_layer_metrics(stark_df, 'starknet')
            
            # Determine optimal routing strategy
            routing_strategy = self._determine_routing_strategy(eth_metrics, stark_metrics)
            
            return {
                'ethereum_metrics': eth_metrics,
                'starknet_metrics': stark_metrics,
                'routing_strategy': routing_strategy,
                'estimated_savings': self._calculate_potential_savings(routing_strategy)
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing cross-layer routing: {str(e)}")
            return {}

    def _calculate_layer_metrics(self, df: pd.DataFrame, layer: str) -> Dict:
        """Calculate performance metrics for a specific layer."""
        try:
            return {
                'average_cost': df['gas_price'].mean() if layer == 'ethereum' else df['fee'].mean(),
                'average_confirmation_time': df['confirmation_time'].mean(),
                'success_rate': len(df[df['status'] == 'success']) / len(df),
                'throughput': len(df) / (df['timestamp'].max() - df['timestamp'].min()),
                'congestion_level': self._calculate_congestion_level(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating layer metrics: {str(e)}")
            return {}

    def _calculate_congestion_level(self, df: pd.DataFrame) -> float:
        """Calculate network congestion level."""
        try:
            # Calculate transactions per minute
            df['minute'] = pd.to_datetime(df['timestamp'], unit='s').dt.floor('min')
            tx_per_minute = df.groupby('minute').size()
            
            # Calculate congestion level based on transaction density
            max_capacity = tx_per_minute.quantile(0.95)  # Assume 95th percentile is max capacity
            current_load = tx_per_minute.mean()
            
            return float(current_load / max_capacity)
            
        except Exception as e:
            self.logger.error(f"Error calculating congestion level: {str(e)}")
            return 0.0

    def _determine_routing_strategy(self, eth_metrics: Dict, stark_metrics: Dict) -> Dict:
        """Determine optimal transaction routing strategy."""
        try:
            # Define routing rules based on transaction characteristics
            routing_rules = []
            
            # Rule 1: High-value transactions
            if eth_metrics['success_rate'] > 0.98:
                routing_rules.append({
                    'condition': 'value > 100 ETH',
                    'preferred_layer': 'ethereum',
                    'reason': 'High security for large transactions'
                })
            
            # Rule 2: Small, frequent transactions
            if stark_metrics['average_cost'] < eth_metrics['average_cost'] * 0.5:
                routing_rules.append({
                    'condition': 'value < 1 ETH AND frequency > 10/day',
                    'preferred_layer': 'starknet',
                    'reason': 'Cost-effective for small transactions'
                })
            
            # Rule 3: Congestion-based routing
            eth_congestion = eth_metrics['congestion_level']
            stark_congestion = stark_metrics['congestion_level']
            
            if eth_congestion > 0.8 and stark_congestion < 0.5:
                routing_rules.append({
                    'condition': 'time_sensitive = True',
                    'preferred_layer': 'starknet',
                    'reason': 'Lower congestion for time-sensitive transactions'
                })
            
            return {
                'rules': routing_rules,
                'default_layer': 'ethereum' if eth_metrics['success_rate'] > stark_metrics['success_rate'] else 'starknet',
                'conditions': self._generate_routing_conditions(eth_metrics, stark_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error determining routing strategy: {str(e)}")
            return {}

    def _generate_routing_conditions(self, eth_metrics: Dict, stark_metrics: Dict) -> List[Dict]:
        """Generate specific conditions for transaction routing."""
        try:
            conditions = []
            
            # Cost-based conditions
            if eth_metrics['average_cost'] > stark_metrics['average_cost'] * 2:
                conditions.append({
                    'name': 'cost_optimization',
                    'threshold': f"value < {eth_metrics['average_cost']} ETH",
                    'action': 'route_to_starknet'
                })
            
            # Speed-based conditions
            if eth_metrics['average_confirmation_time'] > stark_metrics['average_confirmation_time']:
                conditions.append({
                    'name': 'speed_optimization',
                    'threshold': 'urgent = True',
                    'action': 'route_to_starknet'
                })
            
            # Security-based conditions
            conditions.append({
                'name': 'security_optimization',
                'threshold': f"value > 100 ETH",
                'action': 'route_to_ethereum'
            })
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error generating routing conditions: {str(e)}")
            return []

    def _calculate_potential_savings(self, routing_strategy: Dict) -> Dict:
        """Calculate potential savings from optimal routing."""
        try:
            # Initialize savings calculation
            savings = {
                'gas_savings': 0.0,
                'time_savings': 0.0,
                'cost_savings_percentage': 0.0
            }
            
            # Calculate potential savings based on historical data
            if self.historical_data:
                current_costs = sum(tx.get('gas_used', 0) * tx.get('gas_price', 0) 
                                  for tx in self.historical_data)
                
                # Estimate optimized costs
                optimized_costs = self._estimate_optimized_costs(
                    self.historical_data, 
                    routing_strategy
                )
                
                if current_costs > 0:
                    savings['cost_savings_percentage'] = (
                        (current_costs - optimized_costs) / current_costs * 100
                    )
                    savings['gas_savings'] = current_costs - optimized_costs
                    
            return savings
            
        except Exception as e:
            self.logger.error(f"Error calculating potential savings: {str(e)}")
            return {'gas_savings': 0.0, 'time_savings': 0.0, 'cost_savings_percentage': 0.0}

    def _estimate_optimized_costs(self, transactions: List[Dict], 
                                routing_strategy: Dict) -> float:
        """Estimate costs after applying optimization strategy."""
        try:
            optimized_cost = 0.0
            
            for tx in transactions:
                # Determine optimal route
                optimal_layer = self._get_optimal_layer(tx, routing_strategy)
                
                if optimal_layer == 'starknet':
                    # Estimate StarkNet cost
                    optimized_cost += self._estimate_stark_cost(tx)
                else:
                    # Use actual Ethereum cost
                    optimized_cost += tx.get('gas_used', 0) * tx.get('gas_price', 0)
            
            return optimized_cost
            
        except Exception as e:
            self.logger.error(f"Error estimating optimized costs: {str(e)}")
            return 0.0

    def _get_optimal_layer(self, transaction: Dict, routing_strategy: Dict) -> str:
        """Determine optimal layer for a transaction based on routing strategy."""
        try:
            # Check each routing rule
            for rule in routing_strategy.get('rules', []):
                condition = rule['condition']
                
                # Evaluate condition
                if self._evaluate_routing_condition(transaction, condition):
                    return rule['preferred_layer']
            
            # Return default layer if no rules match
            return routing_strategy.get('default_layer', 'ethereum')
            
        except Exception as e:
            self.logger.error(f"Error determining optimal layer: {str(e)}")
            return 'ethereum'

    def _evaluate_routing_condition(self, transaction: Dict, condition: str) -> bool:
        """Evaluate a routing condition for a transaction."""
        try:
            # Parse condition string
            parts = condition.split()
            if len(parts) < 3:
                return False
            
            field, operator, value = parts[0], parts[1], parts[2]
            
            # Get transaction field value
            tx_value = transaction.get(field, 0)
            
            # Convert value to appropriate type
            if 'ETH' in value:
                value = float(value.replace('ETH', ''))
            elif value.replace('.', '').isdigit():
                value = float(value)
            
            # Evaluate condition
            if operator == '>':
                return tx_value > value
            elif operator == '<':
                return tx_value < value
            elif operator == '=':
                return tx_value == value
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating routing condition: {str(e)}")
            return False

    def _estimate_stark_cost(self, transaction: Dict) -> float:
        """Estimate the cost of a transaction if it were on StarkNet."""
        try:
            # Base cost estimation
            base_cost = transaction.get('gas_used', 0) * 0.1  # Assume StarkNet is 90% cheaper
            
            # Adjust for transaction complexity
            complexity_factor = self._calculate_complexity_factor(transaction)
            
            return base_cost * complexity_factor
            
        except Exception as e:
            self.logger.error(f"Error estimating StarkNet cost: {str(e)}")
            return 0.0

    def _calculate_complexity_factor(self, transaction: Dict) -> float:
        """Calculate complexity factor for cost estimation."""
        try:
            # Base complexity
            complexity = 1.0
            
            # Adjust for input data size
            if 'input_data' in transaction:
                data_length = len(transaction['input_data'])
                complexity += 0.1 * (data_length // 1000)
            
            # Adjust for contract interactions
            if transaction.get('to', '').startswith('0x'):
                complexity += 0.2
            
            return min(complexity, 3.0)  # Cap at 3x base cost
            
        except Exception as e:
            self.logger.error(f"Error calculating complexity factor: {str(e)}")
            return 1.0

    def suggest_optimization_strategies(self) -> Dict:
        """Suggest optimization strategies based on analysis."""
        try:
            strategies = {
                'immediate_actions': self._get_immediate_actions(),
                'long_term_recommendations': self._get_long_term_recommendations(),
                'risk_mitigation': self._get_risk_mitigation_strategies()
            }
            
            # Add estimated impact
            strategies['estimated_impact'] = self._calculate_strategy_impact(strategies)
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error suggesting optimization strategies: {str(e)}")
            return {}

    def _get_immediate_actions(self) -> List[Dict]:
        """Get list of immediate optimization actions."""
        try:
            actions = []
            
            # Check gas price optimization
            if self.optimization_results.get('high_gas_costs', False):
                actions.append({
                    'type': 'gas_optimization',
                    'action': 'Shift non-urgent transactions to optimal gas windows',
                    'priority': 'high',
                    'estimated_savings': '20-30%'
                })
            
            # Check layer utilization
            if self.optimization_results.get('layer_imbalance', False):
                actions.append({
                    'type': 'layer_balancing',
                    'action': 'Redistribute transactions across layers based on value and urgency',
                    'priority': 'medium',
                    'estimated_savings': '15-25%'
                })
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Error getting immediate actions: {str(e)}")
            return []

    def _get_long_term_recommendations(self) -> List[Dict]:
        """Get long-term optimization recommendations."""
        try:
            recommendations = []
            
            # Analyze historical trends
            if len(self.historical_data) > 100:
                trend_analysis = self._analyze_historical_trends()
                
                if trend_analysis.get('clear_patterns', False):
                    recommendations.append({
                        'type': 'pattern_optimization',
                        'action': 'Implement predictive transaction scheduling',
                        'implementation_time': '2-3 weeks',
                        'expected_benefit': 'Reduced transaction costs and improved success rates'
                    })
            
            # Infrastructure recommendations
            recommendations.append({
                'type': 'infrastructure',
                'action': 'Implement parallel transaction processing',
                'implementation_time': '1-2 months',
                'expected_benefit': 'Increased throughput and reduced latency'
            })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting long-term recommendations: {str(e)}")
            return []

    def _get_risk_mitigation_strategies(self) -> List[Dict]:
        """Get risk mitigation strategies."""
        try:
            strategies = []
            
            # Transaction monitoring
            strategies.append({
                'type': 'monitoring',
                'strategy': 'Implement real-time transaction monitoring',
                'risk_reduction': 'High',
                'implementation_complexity': 'Medium'
            })
            
            # Fallback mechanisms
            strategies.append({
                'type': 'fallback',
                'strategy': 'Implement cross-layer fallback mechanisms',
                'risk_reduction': 'Medium',
                'implementation_complexity': 'High'
            })
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error getting risk mitigation strategies: {str(e)}")
            return []

    def _calculate_strategy_impact(self, strategies: Dict) -> Dict:
        """Calculate potential impact of suggested strategies."""
        try:
            impact = {
                'cost_reduction': 0.0,
                'performance_improvement': 0.0,
                'risk_reduction': 0.0
            }
            
            # Calculate cost reduction
            for action in strategies.get('immediate_actions', []):
                if 'estimated_savings' in action:
                    savings = float(action['estimated_savings'].split('-')[0])
                    impact['cost_reduction'] += savings
            
            # Estimate performance improvement
            for rec in strategies.get('long_term_recommendations', []):
                if rec['type'] == 'infrastructure':
                    impact['performance_improvement'] += 30.0  # Estimated 30% improvement
            
            # Estimate risk reduction
            for strategy in strategies.get('risk_mitigation', []):
                if strategy['risk_reduction'] == 'High':
                    impact['risk_reduction'] += 40.0
                elif strategy['risk_reduction'] == 'Medium':
                    impact['risk_reduction'] += 20.0
            
            return impact
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy impact: {str(e)}")
            return {'cost_reduction': 0.0, 'performance_improvement': 0.0, 'risk_reduction': 0.0}