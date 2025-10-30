"""
Streamlit UI for Pod Failure Prediction
"""
import streamlit as st
import requests
import json
import pandas as pd

# Page config
st.set_page_config(
    page_title="Pod Failure Predictor",
    page_icon="🚀",
    layout="wide"
)

# Title and description
st.title("🚀 Kubernetes Pod Failure Predictor")
st.markdown("Predict if a pod is at risk of failure based on its metrics")

# Sidebar for API configuration
st.sidebar.header("API Configuration")
api_url = st.sidebar.text_input(
    "API Endpoint", 
    value="http://127.0.0.1:8000/predict",
    help="URL of the FastAPI prediction endpoint"
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Pod Metrics Input")
    
    # Create input form
    with st.form("prediction_form"):
        # Pod identification
        pod_id = st.text_input("Pod ID", value="pod-test-1", help="Unique identifier for the pod")
        
        # Resource metrics
        st.subheader("Resource Metrics")
        col_a, col_b = st.columns(2)
        
        with col_a:
            cpu_usage = st.slider("CPU Usage (%)", 0.0, 100.0, 88.5, 0.1)
            memory_usage = st.slider("Memory Usage (%)", 0.0, 100.0, 95.2, 0.1)
            memory_leak_rate = st.slider("Memory Leak Rate", 0.0, 1.0, 0.22, 0.01)
            
        with col_b:
            restart_count = st.number_input("Restart Count (24h)", 0, 50, 3)
            error_log_rate = st.number_input("Error Log Rate", 0, 100, 10)
            request_latency = st.slider("Request Latency (ms)", 0.0, 1000.0, 180.0, 1.0)
        
        # Scaling and deployment metrics
        st.subheader("Scaling & Deployment Metrics")
        col_c, col_d = st.columns(2)
        
        with col_c:
            replica_count = st.number_input("Replica Count", 1, 20, 4)
            node_pressure = st.slider("Node Pressure Score", 0.0, 1.0, 0.74, 0.01)
            autoscaler_action = st.selectbox(
                "Autoscaler Action", 
                ["none", "scale_up", "scale_down"], 
                index=1
            )
            
        with col_d:
            anomaly_score = st.slider("Prometheus Anomaly Score", 0.0, 1.0, 0.82, 0.01)
            previous_failures = st.number_input("Previous Failures", 0, 20, 2)
            uptime_hours = st.slider("Deployment Uptime (hours)", 0.0, 500.0, 48.0, 1.0)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Pod Failure", use_container_width=True)

with col2:
    st.header("Prediction Results")
    
    if submitted:
        # Prepare the payload
        payload = {
            "pod_id": pod_id,
            "cpu_usage_pct": cpu_usage,
            "memory_usage_pct": memory_usage,
            "memory_leak_rate": memory_leak_rate,
            "restart_count_24h": restart_count,
            "error_log_rate": error_log_rate,
            "request_latency_ms": request_latency,
            "replica_count": replica_count,
            "node_pressure_score": node_pressure,
            "autoscaler_action": autoscaler_action,
            "prometheus_anomaly_score": anomaly_score,
            "previous_failures": previous_failures,
            "deployment_uptime_hrs": uptime_hours
        }
        
        try:
            # Make API request
            with st.spinner("Making prediction..."):
                response = requests.post(
                    api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display prediction
                failure_prob = result["failure_probability"]
                prediction = result["prediction"]
                
                # Color-coded result
                if prediction == "at_risk":
                    st.error(f"⚠️ **Pod Status: AT RISK**")
                    st.error(f"Failure Probability: **{failure_prob:.1%}**")
                else:
                    st.success(f"✅ **Pod Status: STABLE**")
                    st.success(f"Failure Probability: **{failure_prob:.1%}**")
                
                # Progress bar for probability
                st.progress(failure_prob)
                
                # Risk assessment
                st.subheader("Risk Assessment")
                if failure_prob > 0.8:
                    st.error("🔴 **Critical Risk** - Immediate attention required")
                elif failure_prob > 0.6:
                    st.warning("🟡 **Medium Risk** - Monitor closely")
                else:
                    st.success("🟢 **Low Risk** - Pod appears healthy")
                
                # Recommendations
                st.subheader("Recommendations")
                recommendations = []
                
                if cpu_usage > 90:
                    recommendations.append("• Consider scaling up replicas due to high CPU usage")
                if memory_usage > 90:
                    recommendations.append("• Check for memory leaks or increase memory limits")
                if restart_count > 5:
                    recommendations.append("• Investigate frequent restarts - check logs")
                if error_log_rate > 20:
                    recommendations.append("• High error rate detected - review application logs")
                if request_latency > 500:
                    recommendations.append("• High latency detected - optimize application performance")
                
                if recommendations:
                    for rec in recommendations:
                        st.write(rec)
                else:
                    st.write("• Pod metrics look healthy - continue monitoring")
                
                # Show raw response in expander
                with st.expander("View Raw API Response"):
                    st.json(result)
                    
            else:
                st.error(f"API Error: {response.status_code}")
                st.error(response.text)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {str(e)}")
            st.info("Make sure the FastAPI server is running at the specified endpoint")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")

# Footer with example data
st.markdown("---")
st.subheader("Quick Test Examples")

col_ex1, col_ex2, col_ex3 = st.columns(3)

with col_ex1:
    if st.button("🟢 Healthy Pod Example"):
        st.experimental_set_query_params(
            cpu=45.0, memory=60.0, restarts=0, errors=2
        )

with col_ex2:
    if st.button("🟡 Warning Pod Example"):
        st.experimental_set_query_params(
            cpu=75.0, memory=85.0, restarts=2, errors=8
        )

with col_ex3:
    if st.button("🔴 Critical Pod Example"):
        st.experimental_set_query_params(
            cpu=95.0, memory=98.0, restarts=5, errors=25
        )

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Configure API Endpoint**: Set the URL of your FastAPI prediction service in the sidebar
    2. **Enter Pod Metrics**: Fill in the current metrics for your pod
    3. **Get Prediction**: Click the predict button to get failure probability and recommendations
    4. **Review Results**: Check the risk assessment and follow the recommendations
    
    **Metric Descriptions:**
    - **CPU/Memory Usage**: Current resource utilization percentages
    - **Memory Leak Rate**: Rate of memory growth over time
    - **Restart Count**: Number of pod restarts in the last 24 hours
    - **Error Log Rate**: Number of errors per minute in logs
    - **Request Latency**: Average response time in milliseconds
    - **Node Pressure Score**: Pressure on the node hosting the pod
    - **Anomaly Score**: Prometheus-based anomaly detection score
    """)