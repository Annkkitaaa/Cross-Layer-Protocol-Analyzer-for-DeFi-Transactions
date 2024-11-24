# Cross-Layer Protocol Analyzer for DeFi Transactions

A comprehensive system for analyzing and monitoring transactions across Ethereum and Starknet layers, ensuring security, performance, and compliance for decentralized finance (DeFi) applications. This project combines real-time monitoring, cryptographic verification, and machine learning-based anomaly detection to provide insights and optimization strategies for cross-layer DeFi operations.

## 🌟 Features

### Data Extraction and Processing
- Real-time transaction monitoring on Ethereum and Starknet
- Integration with major blockchain APIs (Infura, Alchemy)
- Efficient data storage and processing using SQL and BigQuery

### Security and Verification
- Cryptographic validation of transactions
- Zero-knowledge proof verification
- Signature verification for transaction authenticity

### Analytics and Optimization
- Machine learning-based anomaly detection
- Cross-layer transaction routing optimization
- Gas cost analysis and optimization
- Performance bottleneck identification

### Visualization and Monitoring
- Interactive Streamlit dashboard
- Real-time metrics and alerts
- Cross-layer transaction analysis
- Custom reporting and insights

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Docker (optional)
- Infura API key for Ethereum access
- Starknet node access

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cross-layer-defi-analyzer.git
cd cross-layer-defi-analyzer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export INFURA_URL="your_infura_url"
export STARKNET_NODE="your_starknet_node"
```

### Running the Application

#### Using Python
```bash
streamlit run src/visualization/app.py
```

#### Using Docker
```bash
docker build -t defi-analyzer .
docker run -p 8501:8501 defi-analyzer
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure
```
cross_layer_defi_analyzer/
│
├── data/                  # Data directory
│   ├── raw/              # Raw transaction data
│   └── processed/        # Processed analytics data
│
├── src/                  # Source code
│   ├── data_extraction/  # Data extraction modules
│   │   ├── ethereum_api.py
│   │   └── starknet_api.py
│   │
│   ├── transaction_monitoring/  # Monitoring and crypto
│   │   ├── monitoring.py
│   │   └── cryptography.py
│   │
│   ├── analytics/       # Analysis modules
│   │   ├── anomaly_detection.py
│   │   └── defi_optimization.py
│   │
│   └── visualization/   # Streamlit dashboard
│       └── app.py
│
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| INFURA_URL | Infura API endpoint | Yes |
| STARKNET_NODE | Starknet node URL | Yes |
| LOG_LEVEL | Logging level (default: INFO) | No |

### Application Configuration
The application can be configured through `config.yaml` in the root directory:
```yaml
monitoring:
  update_interval: 60  # seconds
  batch_size: 100     # transactions per batch

analytics:
  anomaly_threshold: 0.8
  optimization_window: 24  # hours
```

## 🔍 Features in Detail

### Anomaly Detection
The system uses machine learning to detect various types of anomalies:
- Unusual transaction patterns
- Potential flash loan attacks
- Abnormal gas usage
- Suspicious contract interactions

### Cross-Layer Optimization
Implements intelligent routing strategies:
- Gas cost optimization
- Transaction speed optimization
- Security level requirements
- Cross-layer communication efficiency

### Real-time Monitoring
Provides continuous monitoring of:
- Transaction status and confirmations
- Gas prices and trends
- Network congestion levels
- Cross-layer state synchronization

## 📧 Contact

Your Name - ankitasingh15.102@gmail.com

Project Link: [https://github.com/Annkkitaaa/cross-layer-defi-analyzer](https://github.com/Annkkitaaa/cross-layer-defi-analyzer)

