# 🎨 Deploy Streamlit UI to Render.com

Since you already have the FastAPI endpoint deployed, here's how to deploy the Streamlit frontend.

## 📋 Prerequisites

- ✅ FastAPI endpoint already deployed on Render
- ✅ Code pushed to GitHub repository
- ✅ Render.com account ready

## 🚀 Deployment Steps

### Step 1: Create Streamlit Web Service

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **New Web Service**: Click "New" → "Web Service"
3. **Connect Repository**: Select your GitHub repository
4. **Configure Service**:

```
Name: pod-failure-ui
Environment: Python 3
Region: Choose your preferred region
Branch: main (or your default branch)
Root Directory: . (leave empty if code is in root)
Build Command: pip install -r requirements.txt
Start Command: python start_streamlit.py
```

### Step 2: Environment Variables

Add these environment variables:

```
PYTHON_VERSION = 3.11.0
RENDER = true
```

### Step 3: Deploy

1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes)
3. Note your Streamlit URL: `https://pod-failure-ui.onrender.com`

### Step 4: Update API Configuration

After deployment, update the API endpoint:

#### Option A: Update config.py (Recommended)

1. **Get your FastAPI URL** from your first deployment
2. **Update config.py** with your actual API URL:

```bash
python update_config.py https://your-api-name.onrender.com/predict
```

3. **Commit and push** the updated config.py:

```bash
git add config.py
git commit -m "Update API endpoint for production"
git push
```

4. **Redeploy**: Render will automatically redeploy when you push

#### Option B: Manual Configuration

1. Visit your Streamlit app
2. In the sidebar, update the "API Endpoint" field
3. Enter your FastAPI URL: `https://your-api-name.onrender.com/predict`

## 🧪 Testing Your Deployment

### 1. Test the Streamlit UI

Visit your Streamlit URL and:
- ✅ Check that the app loads
- ✅ Verify the API endpoint is correct in the sidebar
- ✅ Try the "Healthy Pod Example" button
- ✅ Submit a prediction and verify results

### 2. Test API Connection

The Streamlit app should show:
- ✅ Green success message for healthy pods
- ✅ Red warning for at-risk pods
- ✅ Recommendations based on metrics

## 🔧 Troubleshooting

### Common Issues

**1. "Connection Error" in Streamlit**
- ✅ Check API endpoint URL is correct
- ✅ Ensure FastAPI service is running
- ✅ Verify CORS is enabled in FastAPI

**2. "Build Failed" on Render**
- ✅ Check requirements.txt includes all dependencies
- ✅ Verify Python version compatibility
- ✅ Check build logs for specific errors

**3. "App Not Loading"**
- ✅ Check start command: `python start_streamlit.py`
- ✅ Verify PORT environment variable
- ✅ Check service logs in Render dashboard

### Debug Steps

1. **Check Service Logs**:
   - Go to Render Dashboard → Your Service → Logs
   - Look for startup errors or runtime issues

2. **Test API Independently**:
   ```bash
   curl -X POST 'https://your-api-name.onrender.com/predict' \
     -H 'Content-Type: application/json' \
     -d @example.json
   ```

3. **Local Testing**:
   ```bash
   # Test locally first
   streamlit run streamlit_app.py
   ```

## 🎯 Final URLs

After successful deployment:

- **🎨 Streamlit UI**: `https://pod-failure-ui.onrender.com`
- **🚀 FastAPI Backend**: `https://your-api-name.onrender.com`
- **📚 API Docs**: `https://your-api-name.onrender.com/docs`

## 🔄 Updates and Maintenance

### Automatic Deployments
- Render automatically redeploys when you push to GitHub
- Both services will update independently

### Manual Redeployment
- Go to Render Dashboard → Service → "Manual Deploy"
- Choose "Deploy latest commit"

### Monitoring
- Check service health in Render dashboard
- Monitor logs for errors
- Set up notifications for service failures

## 💡 Pro Tips

1. **Free Tier Limitations**:
   - Services sleep after 15 minutes of inactivity
   - First request after sleep may be slow (cold start)

2. **Performance**:
   - Consider upgrading to paid plans for production
   - Use caching in Streamlit for better performance

3. **Security**:
   - API endpoints are public by default
   - Consider adding authentication for production use

Your pod failure prediction system is now fully deployed! 🎉