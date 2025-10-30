# Deployment Guide for Render.com

This guide explains how to deploy both the FastAPI backend and Streamlit frontend to Render.com.

## Prerequisites

1. **GitHub Repository**: Push your code to a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **Model Files**: Ensure `pod_failure_predictor.pkl` and `preprocessor.pkl` are in your repo

## Option 1: Deploy Both Services (Recommended)

### Step 1: Deploy FastAPI Backend

1. **Create Web Service**:
   - Go to Render Dashboard → "New" → "Web Service"
   - Connect your GitHub repository
   - Configure the service:

```
Name: pod-failure-api
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: python start_api.py
```

2. **Environment Variables**:
   - `PYTHON_VERSION`: `3.11.0`

3. **Deploy**: Click "Create Web Service"
4. **Note the URL**: Save the deployed API URL (e.g., `https://pod-failure-api.onrender.com`)

### Step 2: Deploy Streamlit Frontend

1. **Create Another Web Service**:
   - Go to Render Dashboard → "New" → "Web Service"
   - Connect the same GitHub repository
   - Configure the service:

```
Name: pod-failure-ui
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: python start_streamlit.py
```

2. **Environment Variables**:
   - `PYTHON_VERSION`: `3.11.0`
   - `API_ENDPOINT`: `https://your-api-url.onrender.com/predict` (from Step 1)

3. **Deploy**: Click "Create Web Service"

### Step 3: Update Streamlit Configuration

After deployment, update the default API endpoint in `streamlit_app.py`:

```python
api_url = st.sidebar.text_input(
    "API Endpoint", 
    value="https://your-api-url.onrender.com/predict",  # Update this
    help="URL of the FastAPI prediction endpoint"
)
```

## Option 2: Deploy Using render.yaml (Infrastructure as Code)

1. **Use the provided render.yaml**:
   - The `render.yaml` file in the repository defines both services
   - Push it to your GitHub repository

2. **Deploy from Dashboard**:
   - Go to Render Dashboard → "New" → "Blueprint"
   - Connect your repository
   - Render will automatically create both services

## Option 3: Deploy with Docker

### FastAPI Service
```dockerfile
# Use the existing Dockerfile
# Deploy as a Web Service with Docker environment
```

### Streamlit Service
```dockerfile
# Use Dockerfile.streamlit
# Deploy as a Web Service with Docker environment
```

## Testing Your Deployment

### Test FastAPI Backend
```bash
curl -X POST 'https://your-api-url.onrender.com/predict' \
  -H 'Content-Type: application/json' \
  -d '{
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
  }'
```

### Test Streamlit Frontend
- Visit: `https://your-streamlit-url.onrender.com`
- Update the API endpoint in the sidebar
- Test with the provided examples

## Important Notes

### Free Tier Limitations
- **Cold Starts**: Services sleep after 15 minutes of inactivity
- **Build Time**: Initial deployment may take 5-10 minutes
- **Resource Limits**: 512MB RAM, shared CPU

### Production Considerations
- **Paid Plans**: Consider upgrading for production workloads
- **Environment Variables**: Use Render's environment variables for sensitive data
- **Health Checks**: Both services include basic health checks
- **CORS**: FastAPI includes CORS middleware for cross-origin requests

### Troubleshooting

1. **Build Failures**:
   - Check that all model files are in the repository
   - Verify requirements.txt includes all dependencies
   - Check Python version compatibility

2. **Runtime Errors**:
   - Check service logs in Render dashboard
   - Verify environment variables are set correctly
   - Ensure API endpoint URLs are correct

3. **Connection Issues**:
   - Verify both services are deployed and running
   - Check CORS settings if making cross-origin requests
   - Test API endpoint independently first

## Local Testing Before Deployment

```bash
# Test FastAPI locally
python start_api.py

# Test Streamlit locally (in another terminal)
python start_streamlit.py
```

## Monitoring and Maintenance

- **Logs**: Monitor service logs in Render dashboard
- **Metrics**: Check service metrics and performance
- **Updates**: Redeploy when you push changes to GitHub
- **Scaling**: Consider upgrading plans for higher traffic

## Cost Estimation

- **Free Tier**: Both services can run on free tier
- **Starter Plan**: $7/month per service for better performance
- **Pro Plan**: $25/month per service for production workloads

## Security Best Practices

1. **Environment Variables**: Store sensitive data in environment variables
2. **HTTPS**: Render provides HTTPS by default
3. **Input Validation**: FastAPI includes request validation
4. **Rate Limiting**: Consider adding rate limiting for production

Your pod failure prediction system will be accessible at:
- **API**: `https://pod-failure-api.onrender.com`
- **UI**: `https://pod-failure-ui.onrender.com`