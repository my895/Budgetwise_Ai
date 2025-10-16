#!/usr/bin/env python3
"""
Budget Tool MVP - Enhanced Streamlit Dashboard
Interactive and aesthetic dashboard for budget analysis and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import json
from datetime import datetime, date, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from src.components.category_standardizer import standardizer

# Page configuration
st.set_page_config(
    page_title="Budget Tool MVP",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: none;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Chart containers */
    .plotly-chart {
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        overflow: hidden;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load preprocessed data"""
    data_path = Path("Checkpoints/logs/data_preprocessed.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

@st.cache_data
def load_evaluation_results():
    """Load model evaluation results"""
    results_path = Path("Checkpoints/evaluation/evaluation_results.json")
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def load_model(model_name):
    """Load a trained model"""
    model_dir = Path("Checkpoints/models")
    
    try:
        if model_name == "baseline":
            from src.models.baseline import BaselineModel
            model = BaselineModel()
            model.load(model_dir / "baseline_model.pkl")
            return model
        elif model_name == "random_forest":
            from src.models.ml_model import MLModel
            model = MLModel('random_forest')
            model.load(model_dir / "random_forest_model.pkl")
            return model
        elif model_name == "linear_regression":
            from src.models.ml_model import MLModel
            model = MLModel('linear_regression')
            model.load(model_dir / "linear_regression_model.pkl")
            return model
        elif model_name == "deep_learning":
            from src.models.dl_model import DLModel
            model = DLModel()
            model.load(model_dir / "deep_learning_model.h5")
            return model
    except Exception as e:
        st.error(f"Error loading {model_name} model: {e}")
        return None

def main():
    """Enhanced main dashboard function"""
    
    # Enhanced header with animation
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">💰 Budget Tool MVP Dashboard</h1>
        <p style="font-size: 1.2rem; color: #666; margin-top: -1rem;">
            AI-Powered Financial Analytics & Predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with progress indicators
    with st.spinner("🔄 Loading financial data..."):
        data = load_data()
        eval_results = load_evaluation_results()
    
    if data is None:
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ No preprocessed data found</h4>
            <p>Please run the data preprocessing pipeline first.</p>
            <p><strong>Command:</strong> <code>python setup.py --all-deps</code></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Enhanced sidebar with better styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white; font-family: 'Inter', sans-serif;">📊 Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "Choose a page:",
            ["📈 Data Overview", "🎯 Model Predictions", "📊 Model Performance", "🔧 Settings"],
            help="Select a page to explore different aspects of your financial data"
        )
        
        # Add data summary in sidebar
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Records", len(data))
        st.metric("Categories", data['category'].nunique())
        st.metric("Total Amount", f"${data['amount'].sum():,.0f}")
        
        # Add model status
        st.markdown("---")
        st.markdown("### 🤖 Model Status")
        model_dir = Path("Checkpoints/models")
        models_available = 0
        if (model_dir / "baseline_model.pkl").exists():
            models_available += 1
            st.success("✅ Baseline Model")
        if (model_dir / "random_forest_model.pkl").exists():
            models_available += 1
            st.success("✅ Random Forest")
        if (model_dir / "linear_regression_model.pkl").exists():
            models_available += 1
            st.success("✅ Linear Regression")
        if (model_dir / "deep_learning_model.h5").exists():
            models_available += 1
            st.success("✅ Deep Learning")
        
        if models_available == 0:
            st.warning("⚠️ No models trained")
        else:
            st.info(f"📊 {models_available} models available")
    
    # Route to appropriate page
    if page == "📈 Data Overview":
        show_data_overview(data)
    elif page == "🎯 Model Predictions":
        show_model_predictions(data)
    elif page == "📊 Model Performance":
        show_model_performance(eval_results)
    elif page == "🔧 Settings":
        show_settings()

def show_data_overview(data):
    """Show enhanced data overview page with advanced visualizations"""
    
    # Enhanced header with gradient
    st.markdown('<h2 class="sub-header">📈 Financial Data Overview</h2>', unsafe_allow_html=True)
    
    # Enhanced Summary metrics with custom styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_spending = data['amount'].sum()
    avg_spending = data['amount'].mean()
    max_spending = data['amount'].max()
    min_spending = data['amount'].min()
    num_categories = data['category'].nunique()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(data):,}</p>
            <p class="metric-label">📊 Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${total_spending:,.0f}</p>
            <p class="metric-label">💰 Total Spending</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${avg_spending:,.0f}</p>
            <p class="metric-label">📈 Average Spending</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${max_spending:,.0f}</p>
            <p class="metric-label">⬆️ Highest Amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{num_categories}</p>
            <p class="metric-label">🏷️ Categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced data preview with styling
    st.markdown('<h3 class="sub-header">📋 Data Preview</h3>', unsafe_allow_html=True)
    
    # Add some statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Standard Deviation", f"${data['amount'].std():,.0f}")
    with col2:
        st.metric("Median Amount", f"${data['amount'].median():,.0f}")
    with col3:
        st.metric("Total Transactions", f"{len(data):,}")
    
    # Enhanced dataframe with styling
    styled_data = data.head(10).style.format({
        'amount': '${:,.2f}',
        'date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d')
    }).background_gradient(subset=['amount'], cmap='Blues')
    
    st.dataframe(styled_data, use_container_width=True)
    
    # Category Standardization Info
    st.markdown('<h3 class="sub-header">🏷️ Category Standardization</h3>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ Category Standardization Details", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h4>Categories have been standardized to merge similar variations:</h4>
            <ul>
                <li><strong>RENT</strong>: rent, Rnt, Rentt, RENTT</li>
                <li><strong>FOOD</strong>: food, Fod, Foods, Foodd</li>
                <li><strong>TRAVEL</strong>: travel, Traval, Travl</li>
                <li><strong>ENTERTAINMENT</strong>: entertainment, Entertain, Entrtnmnt</li>
                <li><strong>UTILITIES</strong>: utilities, Utility, Utlities, Utilties</li>
                <li><strong>EDUCATION</strong>: education, EDU, Educaton</li>
                <li><strong>HEALTH</strong>: health, Helth</li>
                <li><strong>SAVINGS</strong>: savings, Saving</li>
                <li><strong>OTHERS</strong>: others, Other, Misc</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced spending analysis
    st.markdown('<h3 class="sub-header">📊 Advanced Spending Analysis</h3>', unsafe_allow_html=True)
    
    category_spending = data.groupby('category')['amount'].sum().sort_values(ascending=False)
    
    # Create a comprehensive dashboard with multiple charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced horizontal bar chart
        fig = go.Figure(data=go.Bar(
            x=category_spending.values,
            y=category_spending.index,
            orientation='h',
            marker=dict(
                color=category_spending.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Amount ($)")
            ),
            text=[f"${val:,.0f}" for val in category_spending.values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Amount: $%{x:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="💰 Total Spending by Category",
                font=dict(size=20, color='#2c3e50')
            ),
            xaxis_title="Amount ($)",
            yaxis_title="Category",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced donut chart
        fig = go.Figure(data=[go.Pie(
            labels=category_spending.index,
            values=category_spending.values,
            hole=0.4,
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='#FFFFFF', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=12),
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text="📊 Spending Distribution",
                font=dict(size=16, color='#2c3e50')
            ),
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced analytics section
    st.markdown('<h3 class="sub-header">📈 Advanced Analytics</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spending distribution histogram
        fig = px.histogram(
            data, 
            x='amount',
            nbins=20,
            title="📊 Spending Amount Distribution",
            labels={'amount': 'Amount ($)', 'count': 'Frequency'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category comparison
        category_stats = data.groupby('category').agg({
            'amount': ['count', 'mean', 'sum']
        }).round(2)
        category_stats.columns = ['Count', 'Average', 'Total']
        category_stats = category_stats.sort_values('Total', ascending=False)
        
        fig = px.scatter(
            category_stats.reset_index(),
            x='Count',
            y='Average',
            size='Total',
            color='Total',
            hover_name='category',
            title="🎯 Category Analysis: Count vs Average",
            labels={'Count': 'Number of Transactions', 'Average': 'Average Amount ($)'},
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Time-based analysis (enhanced)
    st.markdown('<h3 class="sub-header">📅 Time-based Analysis</h3>', unsafe_allow_html=True)
    
    data['date'] = pd.to_datetime(data['date'])
    monthly_spending = data.groupby(data['date'].dt.to_period('M'))['amount'].sum()
    
    # Enhanced line chart with area
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[str(period) for period in monthly_spending.index],
        y=monthly_spending.values,
        mode='lines+markers',
        name='Monthly Spending',
        line=dict(color='#667eea', width=4),
        marker=dict(size=8, color='#667eea'),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title=dict(
            text="📈 Monthly Spending Trend",
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_model_predictions(data):
    """Show enhanced model predictions page with advanced visualizations"""
    st.markdown('<h2 class="sub-header">🎯 AI-Powered Predictions</h2>', unsafe_allow_html=True)
    
    # Model selection with enhanced styling
    available_models = []
    model_dir = Path("Checkpoints/models")
    
    if (model_dir / "baseline_model.pkl").exists():
        available_models.append("baseline")
    if (model_dir / "random_forest_model.pkl").exists():
        available_models.append("random_forest")
    if (model_dir / "linear_regression_model.pkl").exists():
        available_models.append("linear_regression")
    if (model_dir / "deep_learning_model.h5").exists():
        available_models.append("deep_learning")
    
    if not available_models:
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ No trained models found</h4>
            <p>Please train models first by running the complete pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Enhanced model selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_model = st.selectbox(
            "🤖 Select AI Model:", 
            available_models,
            format_func=lambda x: {
                'baseline': '📊 Baseline Model',
                'random_forest': '🌲 Random Forest',
                'linear_regression': '📈 Linear Regression',
                'deep_learning': '🧠 Deep Learning'
            }.get(x, x)
        )
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h4>📊 Model Information</h4>
            <p><strong>Selected:</strong> {selected_model.replace('_', ' ').title()}</p>
            <p><strong>Status:</strong> Ready for predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load selected model
    model = load_model(selected_model)
    if model is None:
        return
    
    st.markdown("""
    <div class="success-card">
        <h4>✅ Model Loaded Successfully!</h4>
        <p>AI model is ready to make predictions based on your financial data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced prediction interface
    st.markdown('<h3 class="sub-header">🔮 Interactive Prediction Interface</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📝 Input Parameters")
        
        # Category selection with enhanced styling
        categories = sorted(data['category'].unique().tolist())
        selected_category = st.selectbox(
            "🏷️ Category:", 
            categories,
            help="Select the spending category for prediction"
        )
        
        # Transaction type
        transaction_type = st.selectbox(
            "💳 Transaction Type:", 
            ["Expense", "Income"],
            help="Type of financial transaction"
        )
        
        # Date selection
        prediction_date = st.date_input(
            "📅 Prediction Date:", 
            value=date.today(),
            help="Date for which to make the prediction"
        )
        
        # Show category standardization info
        st.markdown(f"""
        <div class="info-card">
            <p><strong>Selected Category:</strong> {selected_category} (standardized)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Model Features")
        
        if selected_model != "baseline":
            month = prediction_date.month
            day = prediction_date.day
            day_of_week = prediction_date.weekday()
            
            # Create a feature visualization
            features_data = {
                'Feature': ['Month', 'Day', 'Day of Week', 'Category'],
                'Value': [month, day, day_of_week, selected_category],
                'Type': ['Temporal', 'Temporal', 'Temporal', 'Categorical']
            }
            
            fig = px.bar(
                features_data,
                x='Feature',
                y='Value',
                color='Type',
                title="🎯 Input Features",
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Baseline model uses category-based averages")
    
    # Enhanced prediction button and results
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Make Prediction", key="predict_button"):
            try:
                # Create input data
                input_data = pd.DataFrame({
                    'date': [prediction_date],
                    'category': [selected_category],
                    'transaction_type': [transaction_type]
                })
                
                # Make prediction
                if selected_model == "baseline":
                    prediction = model.predict([selected_category])[0]
                else:
                    prediction = model.predict(input_data)[0]
                
                # Enhanced result display
                st.markdown(f"""
                <div class="success-card">
                    <h3>🎯 Prediction Result</h3>
                    <p style="font-size: 2rem; font-weight: bold; margin: 1rem 0;">
                        ${prediction:,.2f}
                    </p>
                    <p>Predicted amount for <strong>{selected_category}</strong> on <strong>{prediction_date}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence and insights
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Prediction Insights")
                    st.info("🔍 Based on historical spending patterns")
                    st.info("📈 Uses standardized category data")
                    st.info("🎯 Trained on financial transaction history")
                
                with col2:
                    st.markdown("#### 💡 Recommendations")
                    if prediction > data['amount'].mean():
                        st.warning("⚠️ Higher than average spending predicted")
                        st.info("💡 Consider reviewing budget for this category")
                    else:
                        st.success("✅ Within normal spending range")
                        st.info("💡 Good spending control for this category")
                
            except Exception as e:
                st.markdown(f"""
                <div class="warning-card">
                    <h4>❌ Prediction Failed</h4>
                    <p>Error: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Enhanced batch predictions
    st.markdown('<h3 class="sub-header">📊 Batch Predictions Dashboard</h3>', unsafe_allow_html=True)
    
    if st.button("🔄 Generate All Category Predictions", key="batch_predict"):
        try:
            with st.spinner("🤖 Generating predictions for all categories..."):
                predictions = []
                categories_to_predict = sorted(data['category'].unique())
                
                for cat in categories_to_predict:
                    if selected_model == "baseline":
                        pred = model.predict([cat])[0]
                    else:
                        input_data = pd.DataFrame({
                            'date': [prediction_date],
                            'category': [cat],
                            'transaction_type': ['Expense']
                        })
                        pred = model.predict(input_data)[0]
                    
                    predictions.append({
                        'Category': cat, 
                        'Predicted_Amount': pred,
                        'Date': prediction_date
                    })
                
                pred_df = pd.DataFrame(predictions)
                
                # Enhanced results display
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Styled dataframe
                    styled_pred_df = pred_df.style.format({
                        'Predicted_Amount': '${:,.2f}',
                        'Date': lambda x: x.strftime('%Y-%m-%d')
                    }).background_gradient(subset=['Predicted_Amount'], cmap='Blues')
                    
                    st.dataframe(styled_pred_df, use_container_width=True)
                
                with col2:
                    # Prediction summary chart
                    fig = px.bar(
                        pred_df,
                        x='Category',
                        y='Predicted_Amount',
                        title="📊 Predicted Amounts by Category",
                        color='Predicted_Amount',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        height=400,
                        xaxis_tickangle=-45,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.markdown("#### 📈 Prediction Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predicted", f"${pred_df['Predicted_Amount'].sum():,.0f}")
                with col2:
                    st.metric("Average Prediction", f"${pred_df['Predicted_Amount'].mean():,.0f}")
                with col3:
                    st.metric("Highest Prediction", f"${pred_df['Predicted_Amount'].max():,.0f}")
                with col4:
                    st.metric("Lowest Prediction", f"${pred_df['Predicted_Amount'].min():,.0f}")
                
        except Exception as e:
            st.markdown(f"""
            <div class="warning-card">
                <h4>❌ Batch Prediction Failed</h4>
                <p>Error: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)

def show_model_performance(eval_results):
    """Show enhanced model performance page with advanced visualizations"""
    st.markdown('<h2 class="sub-header">📊 AI Model Performance Analytics</h2>', unsafe_allow_html=True)
    
    if eval_results is None:
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ No evaluation results found</h4>
            <p>Please run model evaluation first to see performance metrics.</p>
            <p><strong>Command:</strong> <code>python src/evaluation/model_evaluation.py</code></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Convert to DataFrame for display
    results_df = pd.DataFrame(list(eval_results.values()))
    results_df = results_df.set_index('model')
    
    # Performance overview cards
    st.markdown('<h3 class="sub-header">📈 Performance Overview</h3>', unsafe_allow_html=True)
    
    # Find best model
    best_model = results_df['r2'].idxmax()
    best_r2 = results_df.loc[best_model, 'r2']
    best_mae = results_df.loc[best_model, 'mae']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(results_df)}</p>
            <p class="metric-label">🤖 Models Evaluated</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{best_r2:.3f}</p>
            <p class="metric-label">🏆 Best R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">${best_mae:,.0f}</p>
            <p class="metric-label">🎯 Best MAE</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{best_model.replace('_', ' ').title()}</p>
            <p class="metric-label">🥇 Best Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced performance metrics table
    st.markdown('<h3 class="sub-header">📊 Detailed Performance Metrics</h3>', unsafe_allow_html=True)
    
    # Create styled metrics table
    metrics_display = results_df.copy()
    metrics_display['mae'] = metrics_display['mae'].apply(lambda x: f"${x:,.2f}")
    metrics_display['mse'] = metrics_display['mse'].apply(lambda x: f"${x:,.2f}")
    metrics_display['rmse'] = metrics_display['rmse'].apply(lambda x: f"${x:,.2f}")
    metrics_display['r2'] = metrics_display['r2'].apply(lambda x: f"{x:.4f}")
    metrics_display['mape'] = metrics_display['mape'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(metrics_display, use_container_width=True)
    
    # Advanced performance visualizations
    st.markdown('<h3 class="sub-header">📈 Performance Comparison Dashboard</h3>', unsafe_allow_html=True)
    
    # Create subplots for comprehensive comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score (Higher is Better)', 'Mean Absolute Error (Lower is Better)', 
                       'Root Mean Square Error (Lower is Better)', 'Mean Absolute Percentage Error (Lower is Better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² Score
    fig.add_trace(
        go.Bar(
            x=results_df.index,
            y=results_df['r2'],
            name='R² Score',
            marker_color='#667eea',
            text=results_df['r2'].round(4),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(
            x=results_df.index,
            y=results_df['mae'],
            name='MAE',
            marker_color='#764ba2',
            text=[f"${val:,.0f}" for val in results_df['mae']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(
            x=results_df.index,
            y=results_df['rmse'],
            name='RMSE',
            marker_color='#f093fb',
            text=[f"${val:,.0f}" for val in results_df['rmse']],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # MAPE
    fig.add_trace(
        go.Bar(
            x=results_df.index,
            y=results_df['mape'],
            name='MAPE',
            marker_color='#4facfe',
            text=[f"{val:.2f}%" for val in results_df['mape']],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        title_text="📊 Comprehensive Model Performance Comparison",
        title_x=0.5,
        title_font_size=20
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model ranking and insights
    st.markdown('<h3 class="sub-header">🏆 Model Rankings & Insights</h3>', unsafe_allow_html=True)
    
    # Create ranking
    results_df_ranked = results_df.copy()
    results_df_ranked['r2_rank'] = results_df_ranked['r2'].rank(ascending=False)
    results_df_ranked['mae_rank'] = results_df_ranked['mae'].rank(ascending=True)
    results_df_ranked['overall_rank'] = (results_df_ranked['r2_rank'] + results_df_ranked['mae_rank']) / 2
    results_df_ranked = results_df_ranked.sort_values('overall_rank')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Ranking visualization
        fig = px.scatter(
            results_df_ranked.reset_index(),
            x='r2',
            y='mae',
            size='overall_rank',
            color='overall_rank',
            hover_name='model',
            title="🎯 Model Performance Scatter Plot",
            labels={'r2': 'R² Score (Higher Better)', 'mae': 'MAE (Lower Better)'},
            color_continuous_scale='RdYlGn_r'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model recommendations
        st.markdown("#### 💡 Model Recommendations")
        
        # Best model info
        st.markdown(f"""
        <div class="success-card">
            <h4>🥇 Best Overall Model</h4>
            <p><strong>{best_model.replace('_', ' ').title()}</strong></p>
            <p>R²: {best_r2:.4f}</p>
            <p>MAE: ${best_mae:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model insights
        st.markdown("#### 📊 Performance Insights")
        
        if best_r2 > 0.8:
            st.success("🎉 Excellent model performance!")
        elif best_r2 > 0.6:
            st.info("👍 Good model performance")
        else:
            st.warning("⚠️ Consider improving model features")
        
        # Model type insights
        model_insights = {
            'baseline': 'Simple category averages - good for quick estimates',
            'random_forest': 'Robust to outliers - good for complex patterns',
            'linear_regression': 'Fast and interpretable - good for linear relationships',
            'deep_learning': 'Complex patterns - needs more data for optimal performance'
        }
        
        for model_name, insight in model_insights.items():
            if model_name in results_df.index:
                st.info(f"**{model_name.title()}:** {insight}")
    
    # Performance trends
    st.markdown('<h3 class="sub-header">📈 Performance Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance radar chart
        categories = ['R² Score', 'MAE (inverted)', 'RMSE (inverted)', 'MAPE (inverted)']
        
        fig = go.Figure()
        
        for model_name in results_df.index:
            # Normalize values for radar chart (0-1 scale)
            r2_norm = results_df.loc[model_name, 'r2']
            mae_norm = 1 - (results_df.loc[model_name, 'mae'] / results_df['mae'].max())
            rmse_norm = 1 - (results_df.loc[model_name, 'rmse'] / results_df['rmse'].max())
            mape_norm = 1 - (results_df.loc[model_name, 'mape'] / results_df['mape'].max())
            
            fig.add_trace(go.Scatterpolar(
                r=[r2_norm, mae_norm, rmse_norm, mape_norm],
                theta=categories,
                fill='toself',
                name=model_name.replace('_', ' ').title()
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="🎯 Model Performance Radar Chart",
            height=400,
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model complexity vs performance
        complexity_scores = {
            'baseline': 1,
            'linear_regression': 2,
            'random_forest': 3,
            'deep_learning': 4
        }
        
        complexity_data = []
        for model_name in results_df.index:
            if model_name in complexity_scores:
                complexity_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Complexity': complexity_scores[model_name],
                    'Performance': results_df.loc[model_name, 'r2']
                })
        
        complexity_df = pd.DataFrame(complexity_data)
        
        fig = px.scatter(
            complexity_df,
            x='Complexity',
            y='Performance',
            size='Performance',
            color='Performance',
            hover_name='Model',
            title="⚖️ Model Complexity vs Performance",
            labels={'Performance': 'R² Score', 'Complexity': 'Model Complexity'},
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_settings():
    """Show settings page"""
    st.header("🔧 Settings")
    
    st.subheader("📁 Data Information")
    
    # Check data files
    data_files = {
        "Raw Data": Path("Data/raw/data.csv"),
        "Preprocessed Data": Path("Checkpoints/logs/data_preprocessed.csv"),
        "Ingested Data": Path("Checkpoints/logs/data_ingested.csv")
    }
    
    for name, path in data_files.items():
        if path.exists():
            st.success(f"✅ {name}: {path}")
        else:
            st.error(f"❌ {name}: {path} (not found)")
    
    st.subheader("🤖 Model Information")
    
    # Check model files
    model_files = {
        "Baseline Model": Path("Checkpoints/models/baseline_model.pkl"),
        "Random Forest": Path("Checkpoints/models/random_forest_model.pkl"),
        "Linear Regression": Path("Checkpoints/models/linear_regression_model.pkl"),
        "Deep Learning": Path("Checkpoints/models/deep_learning_model.h5")
    }
    
    for name, path in model_files.items():
        if path.exists():
            file_size = path.stat().st_size / 1024  # KB
            st.success(f"✅ {name}: {path} ({file_size:.1f} KB)")
        else:
            st.error(f"❌ {name}: {path} (not found)")
    
    st.subheader("📊 Evaluation Information")
    
    eval_path = Path("Checkpoints/evaluation/evaluation_results.json")
    if eval_path.exists():
        st.success(f"✅ Evaluation Results: {eval_path}")
    else:
        st.error(f"❌ Evaluation Results: {eval_path} (not found)")
    
    # Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Run Evaluation"):
            st.info("Run: `python src/evaluation/model_evaluation.py`")
    
    with col3:
        if st.button("🏃 Run Full Pipeline"):
            st.info("Run: `python quick_start.py`")

if __name__ == "__main__":
    main()
