# Network Anomaly Detector

A machine learning-based network traffic analyzer that detects anomalies and potential security threats in real-time.

## Features

- Real-time network traffic monitoring
- Machine learning-based anomaly detection
- Traffic pattern analysis
- Alert system for suspicious activities
- Data visualization for network metrics

## Prerequisites

- Python 3.8+
- Required Python packages:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - scapy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Avishka-Gamage/Network-Anomaly-Detection.git
cd NetworkAnomalyDetector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the network monitoring:
```bash
python app.py
```

2. Access the dashboard at `http://localhost:8080`

## Configuration

Edit `config.yaml` to customize:
- Network interface
- Detection sensitivity
- Alert thresholds
- Logging preferences

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/NetworkAnomalyDetector

## Datasets

This project uses the CIC-IDS-2017 dataset for training and testing the anomaly detection model. The dataset contains benign and the most up-to-date common attacks, which resembles real-world data.

Dataset: [CIC-IDS-2017](http://cicresearch.ca/CICDataset/CIC-IDS-2017/)

Features of the dataset:
- Contains benign and attack network traffic
- Includes various types of attacks (DoS, DDoS, Infiltration, Web Attack, etc.)
- Created with real network traffic data
- Comprehensive flow-based features
