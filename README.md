# Next Pod Failure Prediction

This project trains a binary classifier to predict the next pod failure in a Kubernetes cluster and serves it via FastAPI with a Streamlit web interface.

## 🚀 Features
- **Machine Learning Pipeline**: Automated training with model selection (LogisticRegression, RandomForest, XGBoost)
- **FastAPI Backend**: RESTful API for predictions with automatic documentation
- **Streamlit UI**: Interactive web interface for easy predictions and monitoring
- **Docker Support**: Containerized deployment ready
- **Cloud Deployment**: Ready for Render.com deployment

## 📁 Project Structure
```
├── train_model.py              # ML training pipeline
├── evaluate_model.py           # Model evaluation and testing
├── app.py                      # FastAPI backend service
├── streamlit_app.py            # Streamlit web interface
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container for FastAPI
├── Dockerfile.streamlit        # Docker container for Streamlit
├── render.yaml                 # Render.com deployment config
├── DEPLOYMENT.md               # Detailed deployment guide
└── example.json                # Sample prediction request
```

## 🏃‍♂️ Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Services

#### Option A: FastAPI + Streamlit (Recommended)
```bash
# Terminal 1: Start FastAPI backend
uvicorn app:app --reload --port 8000

# Terminal 2: Start Streamlit UI
streamlit run streamlit_app.py --server.port 8501
```

Then visit:
- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs

#### Option B: API Only
```bash
# Start FastAPI
uvicorn app:app --reload --port 8000

# Test with curl
curl -X POST 'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d @example.json
```

## 🌐 Web Interface

The Streamlit UI provides:
- **Interactive Form**: Easy input of pod metrics
- **Real-time Predictions**: Instant failure probability assessment
- **Risk Assessment**: Color-coded risk levels with recommendations
- **Example Scenarios**: Pre-configured test cases
- **API Configuration**: Flexible backend endpoint configuration

### Using the Web Interface
1. Open http://localhost:8501
2. Configure the API endpoint (default: http://127.0.0.1:8000/predict)
3. Enter pod metrics using the form
4. Click "Predict Pod Failure" to get results
5. Review risk assessment and recommendations

## 🐳 Docker Deployment

### FastAPI Service
```bash
docker build -t pod-failure-api .
docker run -p 8000:8000 pod-failure-api
```

### Streamlit Service
```bash
docker build -f Dockerfile.streamlit -t pod-failure-ui .
docker run -p 8501:8501 pod-failure-ui
```

### Docker Compose
```bash
docker-compose up
```

## ☁️ Cloud Deployment

### Deploy to Render.com
See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Quick Deploy:**
1. Push code to GitHub
2. Connect repository to Render.com
3. Deploy using the provided `render.yaml`
4. Access your services at the provided URLs

## 📊 Model Performance

The training pipeline automatically:
- Compares multiple algorithms (LogisticRegression, RandomForest, XGBoost)
- Performs hyperparameter tuning
- Selects the best model based on ROC-AUC score
- Saves model artifacts and training summary

**Current Performance:**
- Best Model: LogisticRegression
- ROC-AUC: ~0.59 (varies by dataset)
- Features: 11 numerical + 2 categorical

## 🔧 API Reference

### POST /predict
Predict pod failure probability.

**Request Body:**
```json
{
  "pod_id": "pod-test-1",
  "cpu_usage_pct": 88.5,
  "memory_usage_pct": 95.2,
  "memory_leak_rate": 0.22,
  "restart_count_24h": 3,
  "error_log_rate": 10,
  "request_latency_ms": 180,
  "replica_count": 4,
  "node_pressure_score": 0.74,
  "autoscaler_action": "scale_up",
  "prometheus_anomaly_score": 0.82,
  "previous_failures": 2,
  "deployment_uptime_hrs": 48
}
```

**Response:**
```json
{
  "input": {...},
  "failure_probability": 0.9404,
  "prediction": "at_risk"
}
```

## 🛠️ Development

### Adding New Features
1. **New Metrics**: Update the `PodMetrics` model in `app.py` and retrain
2. **New Algorithms**: Add to the `models` dictionary in `train_model.py`
3. **UI Enhancements**: Modify `streamlit_app.py`

### Testing
```bash
# Test model training
python train_model.py

# Test model evaluation
python evaluate_model.py

# Test API
python -m pytest tests/  # (if you add tests)
```

## 📈 Monitoring & Maintenance

- **Model Retraining**: Regularly retrain with new data
- **Performance Monitoring**: Track prediction accuracy over time
- **Resource Monitoring**: Monitor API response times and resource usage
- **Log Analysis**: Review application logs for errors and patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment help
- **API Docs**: Visit `/docs` endpoint when running FastAPI locallyson
```

## Notes
- The training script will detect common target column names (`predicted_pod_failure`, `next_pod_failure`).
- Save and version the `pod_failure_predictor.pkl` artifact and consider CI/CD for retraining on new metrics.