
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List
import sys
import os

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class DeFiAnalyzerDashboard:
    def __init__(self, eth_extractor, stark_extractor, anomaly_detector, defi_optimizer):
        """Initialize the dashboard with required components."""
        self.eth_extractor = eth_extractor
        self.stark_extractor = stark_extractor
        self.anomaly_detector = anomaly_detector
        self.defi_optimizer = defi_optimizer
        
        # Set page config
        st.set_page_config(
            page_title="Cross-Layer DeFi Analyzer",
            page_icon="📊",
            layout="wide"
        )

    def run(self):
        """Run the Streamlit dashboard."""
        st.title("Cross-Layer DeFi Protocol Analyzer")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_transaction_metrics()
            self._render_gas_analysis()
        
        with col2:
            self._render_anomaly_detection()
            self._render_layer_comparison()
        
        # Bottom section
        self._render_optimization_suggestions()

    def _render_sidebar(self):
        """Render sidebar with filters and controls."""
        st.sidebar.header("Controls")
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Last 24 hours", "Last 7 days", "Last 30 days"]
        )
        
        # Layer selector
        selected_layers = st.sidebar.multiselect(
            "Layers",
            ["Ethereum", "StarkNet"],
            default=["Ethereum", "StarkNet"]
        )
        
        # Transaction type filter
        tx_types = st.sidebar.multiselect(
            "Transaction Types",
            ["Swaps", "Liquidity Provision", "Borrowing", "Lending"],
            default=["Swaps"]
        )
        
        # Update button
        if st.sidebar.button("Update Data"):
            st.experimental_rerun()

    def _render_transaction_metrics(self):
        """Render key transaction metrics."""
        st.subheader("Transaction Metrics")
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Transactions",
                value="1,234",
                delta="43%"
            )
        
        with col2:
            st.metric(
                label="Average Gas Price",
                value="45 Gwei",
                delta="-12%"
            )
        
        with col3:
            st.metric(
                label="Success Rate",
                value="98.5%",
                delta="2.1%"
            )
        
        # Transaction volume chart
        tx_data = self._get_transaction_data()
        fig = px.line(
            tx_data,
            x="timestamp",
            y="volume",
            title="Transaction Volume Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

    def _render_gas_analysis(self):
        """Render gas price analysis."""
        st.subheader("Gas Analysis")
        
        # Gas price distribution
        gas_data = self._get_gas_data()
        fig = px.histogram(
            gas_data,
            x="gas_price",
            nbins=50,
            title="Gas Price Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal gas times
        st.subheader("Optimal Gas Times")
        optimal_times = self.defi_optimizer.analyze_gas_efficiency({})
        
        if optimal_times.get('optimal_gas_windows'):
            for window in optimal_times['optimal_gas_windows']:
                st.write(f"🕒 {window['hour']}:00 - {window['hour']+1}:00")
                st.write(f"Average Gas Price: {window['average_gas_price']} Gwei")

    def _render_anomaly_detection(self):
        """Render anomaly detection results."""
        st.subheader("Anomaly Detection")
        
        # Recent anomalies
        anomalies = self._get_recent_anomalies()
        if anomalies:
            for anomaly in anomalies:
                with st.expander(f"Anomaly at {anomaly['timestamp']}"):
                    st.write(f"Type: {anomaly['anomaly_type']}")
                    st.write(f"Score: {anomaly['anomaly_score']:.2f}")
                    st.write(f"Transaction Hash: {anomaly['hash']}")
        else:
            st.info("No anomalies detected in the selected time range")

    def _render_layer_comparison(self):
        """Render layer comparison metrics."""
        st.subheader("Layer Comparison")
        
        # Create comparison metrics
        comparison = self._get_layer_comparison()
        
        # Render comparison chart
        fig = go.Figure(data=[
            go.Bar(
                name='Ethereum',
                x=['Cost', 'Speed', 'Success Rate'],
                y=[comparison['ethereum']['cost'], 
                   comparison['ethereum']['speed'],
                   comparison['ethereum']['success_rate']]
            ),
            go.Bar(
                name='StarkNet',
                x=['Cost', 'Speed', 'Success Rate'],
                y=[comparison['starknet']['cost'],
                   comparison['starknet']['speed'],
                   comparison['starknet']['success_rate']]
            )
        ])
        
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    def _render_optimization_suggestions(self):
        """Render optimization suggestions."""
        st.subheader("Optimization Suggestions")
        
        # Get optimization strategies
        strategies = self.defi_optimizer.suggest_optimization_strategies()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Render immediate actions
            st.write("#### Immediate Actions")
            for action in strategies.get('immediate_actions', []):
                with st.expander(f"🎯 {action['type'].title()}"):
                    st.write(f"**Action:** {action['action']}")
                    st.write(f"**Priority:** {action['priority']}")
                    st.write(f"**Estimated Savings:** {action['estimated_savings']}")
        
        with col2:
            # Render long-term recommendations
            st.write("#### Long-term Recommendations")
            for rec in strategies.get('long_term_recommendations', []):
                with st.expander(f"📈 {rec['type'].title()}"):
                    st.write(f"**Action:** {rec['action']}")
                    st.write(f"**Implementation Time:** {rec['implementation_time']}")
                    st.write(f"**Expected Benefit:** {rec['expected_benefit']}")
        
        # Render impact analysis
        if strategies.get('estimated_impact'):
            st.write("#### Estimated Impact")
            impact = strategies['estimated_impact']
            
            impact_data = pd.DataFrame({
                'Metric': ['Cost Reduction', 'Performance Improvement', 'Risk Reduction'],
                'Percentage': [
                    impact['cost_reduction'],
                    impact['performance_improvement'],
                    impact['risk_reduction']
                ]
            })
            
            fig = px.bar(
                impact_data,
                x='Metric',
                y='Percentage',
                title='Estimated Impact of Optimization Strategies'
            )
            st.plotly_chart(fig, use_container_width=True)

    def _get_transaction_data(self) -> pd.DataFrame:
        """Get transaction data for visualization."""
        try:
            # Sample data - replace with actual data in production
            dates = pd.date_range(start='today', periods=24, freq='H')
            volumes = np.random.normal(1000, 200, 24)
            
            return pd.DataFrame({
                'timestamp': dates,
                'volume': volumes
            })
        except Exception as e:
            st.error(f"Error getting transaction data: {str(e)}")
            return pd.DataFrame()

    def _get_gas_data(self) -> pd.DataFrame:
        """Get gas price data for visualization."""
        try:
            # Sample data - replace with actual data in production
            gas_prices = np.random.gamma(shape=2, scale=20, size=1000)
            return pd.DataFrame({'gas_price': gas_prices})
        except Exception as e:
            st.error(f"Error getting gas data: {str(e)}")
            return pd.DataFrame()

    def _get_recent_anomalies(self) -> List[Dict]:
        """Get recent anomaly detection results."""
        try:
            # Sample data - replace with actual anomaly detection results
            return [
                {
                    'timestamp': datetime.now() - timedelta(minutes=30),
                    'anomaly_type': 'high_value_transaction',
                    'anomaly_score': 0.92,
                    'hash': '0x123...'
                },
                {
                    'timestamp': datetime.now() - timedelta(hours=2),
                    'anomaly_type': 'potential_flash_loan',
                    'anomaly_score': 0.85,
                    'hash': '0x456...'
                }
            ]
        except Exception as e:
            st.error(f"Error getting anomalies: {str(e)}")
            return []

    def _get_layer_comparison(self) -> Dict:
        """Get comparison metrics between layers."""
        try:
            return {
                'ethereum': {
                    'cost': 100,  # Normalized values
                    'speed': 70,
                    'success_rate': 98
                },
                'starknet': {
                    'cost': 20,
                    'speed': 90,
                    'success_rate': 95
                }
            }
        except Exception as e:
            st.error(f"Error getting layer comparison: {str(e)}")
            return {}

if __name__ == "__main__":
    # Initialize components
    from src.data_extraction.ethereum_api import EthereumDataExtractor
    from src.data_extraction.starknet_api import StarknetDataExtractor
    from src.analytics.anomaly_detection import DeFiAnomalyDetector
    from src.analytics.defi_optimization import DeFiOptimizer
    
    # Create instances
    eth_extractor = EthereumDataExtractor(infura_url="YOUR_INFURA_URL")
    stark_extractor = StarknetDataExtractor(node_url="YOUR_STARKNET_NODE")
    anomaly_detector = DeFiAnomalyDetector()
    defi_optimizer = DeFiOptimizer()
    
    # Create and run dashboard
    dashboard = DeFiAnalyzerDashboard(
        eth_extractor=eth_extractor,
        stark_extractor=stark_extractor,
        anomaly_detector=anomaly_detector,
        defi_optimizer=defi_optimizer
    )
    dashboard.run()