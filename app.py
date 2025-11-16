"""
═══════════════════════════════════════════════════════════════════════════
PROCUREAI UAE - COMPREHENSIVE ANALYTICS DASHBOARD
Machine Learning Suite for Customer Segmentation & Dynamic Pricing
═══════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve,
                            silhouette_score, silhouette_samples,
                            r2_score, mean_squared_error, mean_absolute_error, 
                            accuracy_score, precision_score, recall_score, f1_score)
from mlxtend.frequent_patterns import apriori, association_rules

# Set page config
st.set_page_config(
    page_title="ProcureAI Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDownloadButton button {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def fix_dtypes_for_arrow(df):
    """Fix pandas nullable dtypes for Arrow compatibility"""
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        if hasattr(df[col].dtype, 'name'):
            if df[col].dtype.name == 'Int64':
                df[col] = df[col].astype('float64')
            elif df[col].dtype.name == 'boolean':
                df[col] = df[col].astype('bool')
    return df

@st.cache_data
def load_default_data():
    """Load synthetic dataset"""
    # Try to load from GitHub or local
    try:
        df = pd.read_csv("ProcureAI_Survey_Data.csv")
        return df
    except:
        return generate_synthetic_data()

def generate_synthetic_data(n=600):
    """Generate synthetic procurement data"""
    np.random.seed(42)
    
    data = {
        # Firmographics
        'Industry': np.random.choice(['Construction', 'Manufacturing', 'Trading', 'Professional Services',
                                     'Healthcare', 'Retail', 'Technology'], n),
        'Employees': np.random.randint(5, 1000, n),
        'Annual_Revenue_Million_AED': np.random.uniform(0.5, 200, n),
        'Years_Operating': np.random.randint(1, 30, n),
        
        # Procurement Profile
        'Annual_Procurement_Spend_AED': np.random.uniform(100000, 100000000, n),
        'Vendor_Count': np.random.randint(5, 300, n),
        'Procurement_FTE': np.random.randint(0, 10, n),
        
        # Current State
        'Uses_Manual_Process': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'Has_ERP': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'Digital_Maturity_Score': np.random.randint(1, 6, n),
        'Tech_Openness': np.random.randint(1, 6, n),
        
        # Pain Points
        'Pain_Manual_RFQ': np.random.randint(1, 6, n),
        'Pain_Vendor_Risk': np.random.randint(1, 6, n),
        'Pain_Compliance': np.random.randint(1, 6, n),
        
        # Feature Values
        'Values_Automation': np.random.randint(1, 6, n),
        'Values_Risk_Assessment': np.random.randint(1, 6, n),
        'Values_Compliance': np.random.randint(1, 6, n),
        
        # Intent & Budget
        'Interest_Level': np.random.randint(1, 6, n),
        'Purchase_Urgency': np.random.randint(1, 6, n),
        'Max_Monthly_WTP_AED': np.random.uniform(500, 15000, n),
        'Current_Software_Budget_Annual_AED': np.random.uniform(5000, 500000, n),
    }
    
    df = pd.DataFrame(data)
    
    # Create derived features
    df['Avg_Pain_Score'] = df[['Pain_Manual_RFQ', 'Pain_Vendor_Risk', 'Pain_Compliance']].mean(axis=1)
    df['Avg_Feature_Value'] = df[['Values_Automation', 'Values_Risk_Assessment', 'Values_Compliance']].mean(axis=1)
    
    # Create target for classification
    df['Is_Hot_Lead'] = ((df['Interest_Level'] >= 4) & (df['Avg_Pain_Score'] >= 3.5)).astype(int)
    
    # Create cluster placeholder
    df['Cluster'] = -1
    
    # Fix nullable dtypes
    for col in df.select_dtypes(include=['Int64']).columns:
        df[col] = df[col].astype('float64')
    
    return df

def create_download_link(df, filename, link_text):
    """Create download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
               yticklabels=class_names, ax=ax)
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=ProcureAI", width=200)
    st.sidebar.title("🎛️ Navigation")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Data Upload Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Data Management")
    
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Use Synthetic Data", "Upload Custom Data"]
    )
    
    if data_source == "Upload Custom Data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Custom data loaded!")
    else:
        if st.sidebar.button("Load Synthetic Data"):
            st.session_state.data = generate_synthetic_data(600)
            st.sidebar.success("✅ Synthetic data loaded!")
    
    # Navigation
    st.sidebar.markdown("---")
    page = st.sidebar.selectbox(
        "Select Analysis:",
        ["🏠 Home & Overview",
         "🎯 Classification Analysis",
         "👥 Clustering & Segmentation",
         "🔗 Association Rule Mining",
         "📈 Regression & Pricing",
         "💰 Dynamic Pricing Engine"]
    )
    
    # Main content
    if st.session_state.data is None:
        st.markdown("<h1 class='main-header'>🚀 ProcureAI Analytics Dashboard</h1>", unsafe_allow_html=True)
        st.info("👈 Please load data from the sidebar to begin analysis")
        
        st.markdown("""
        ## 🎯 Welcome to ProcureAI Analytics Suite
        
        This comprehensive dashboard provides:
        
        - **🎯 Classification**: Predict customer purchase intent
        - **👥 Clustering**: Segment customers into personas
        - **🔗 Association Rules**: Discover feature patterns
        - **📈 Regression**: Predict willingness to pay
        - **💰 Dynamic Pricing**: Generate personalized pricing recommendations
        
        ### 📊 Getting Started:
        1. Load synthetic data or upload your own dataset
        2. Navigate through different analysis tabs
        3. Adjust parameters using interactive controls
        4. Download results for business use
        """)
        return
    
    df = st.session_state.data.copy()
    
    # Route to appropriate page
    if page == "🏠 Home & Overview":
        show_home(df)
    elif page == "🎯 Classification Analysis":
        show_classification(df)
    elif page == "👥 Clustering & Segmentation":
        show_clustering(df)
    elif page == "🔗 Association Rule Mining":
        show_association_rules(df)
    elif page == "📈 Regression & Pricing":
        show_regression(df)
    elif page == "💰 Dynamic Pricing Engine":
        show_dynamic_pricing(df)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME & OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def show_home(df):
    st.markdown("<h1 class='main-header'>🏠 Data Overview & Summary</h1>", unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Customers", f"{len(df):,}")
    
    with col2:
        if 'Max_Monthly_WTP_AED' in df.columns:
            st.metric("💰 Avg WTP", f"AED {df['Max_Monthly_WTP_AED'].mean():,.0f}")
    
    with col3:
        if 'Is_Hot_Lead' in df.columns:
            hot_leads = df['Is_Hot_Lead'].sum()
            st.metric("🔥 Hot Leads", f"{hot_leads} ({hot_leads/len(df)*100:.1f}%)")
    
    with col4:
        if 'Annual_Procurement_Spend_AED' in df.columns:
            st.metric("💼 Avg Procurement", f"AED {df['Annual_Procurement_Spend_AED'].mean()/1e6:.1f}M")
    
    st.markdown("---")
    
    # Dataset preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(fix_dtypes_for_arrow(df.head(20)), width="stretch")
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Statistics")
        st.write(f"**Rows:** {df.shape[0]:,}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.subheader("🔢 Column Types")
        col_types = df.dtypes.value_counts()
        st.write(col_types)
    
    # Numeric columns distribution
    st.markdown("---")
    st.subheader("📊 Numeric Columns Distribution")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select column to visualize:", numeric_cols)
        
        fig = px.histogram(df, x=selected_col, nbins=50, 
                          title=f"Distribution of {selected_col}",
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download data
    st.markdown("---")
    st.subheader("💾 Download Dataset")
    st.markdown(create_download_link(df, "procureai_data.csv", "📥 Download Current Dataset (CSV)"), 
               unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: CLASSIFICATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def show_classification(df):
    st.markdown("<h1 class='main-header'>🎯 Classification: Predict Purchase Intent</h1>", unsafe_allow_html=True)
    
    # Check if target exists
    if 'Is_Hot_Lead' not in df.columns:
        # Create target variable safely
        if 'Interest_Level' in df.columns and 'Avg_Pain_Score' in df.columns:
            df['Is_Hot_Lead'] = ((df['Interest_Level'] >= 4) & 
                                (df['Avg_Pain_Score'] >= 3.5)).astype(int)
        else:
            # Fallback: create dummy target
            df['Is_Hot_Lead'] = 0
            st.warning("⚠️ Creating dummy target variable. Upload data with Interest_Level and Avg_Pain_Score for better results.")
    
    st.info("🎯 **Objective:** Predict which customers are 'Hot Leads' (ready to buy or highly interested)")
    
    # Feature selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Classification Settings")
    
    # Select features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ['Is_Hot_Lead', 'Cluster', 
                                                                  'Predicted_Hot_Lead', 'Hot_Lead_Probability']]
    
    default_features = [f for f in ['Avg_Pain_Score', 'Interest_Level', 'Purchase_Urgency',
                                    'Max_Monthly_WTP_AED', 'Digital_Maturity_Score', 
                                    'Tech_Openness', 'Annual_Procurement_Spend_AED'] 
                       if f in numeric_features]
    
    if len(default_features) == 0:
        default_features = numeric_features[:7] if len(numeric_features) >= 7 else numeric_features
    
    selected_features = st.sidebar.multiselect(
        "Select Features:",
        numeric_features,
        default=default_features
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features")
        return
    
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 30) / 100
    
    # Prepare data
    X = df[selected_features].fillna(df[selected_features].median())
    y = df['Is_Hot_Lead']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Select Model:",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )
    
    # Train model
    with st.spinner("🔄 Training model..."):
        if model_choice == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_choice == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        acc = accuracy_score(y_test, y_pred)
        st.metric("🎯 Accuracy", f"{acc:.2%}")
    
    with col2:
        prec = precision_score(y_test, y_pred, zero_division=0)
        st.metric("✅ Precision", f"{prec:.2%}")
    
    with col3:
        rec = recall_score(y_test, y_pred, zero_division=0)
        st.metric("🔍 Recall", f"{rec:.2%}")
    
    with col4:
        f1 = f1_score(y_test, y_pred, zero_division=0)
        st.metric("📈 F1-Score", f"{f1:.2%}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = plot_confusion_matrix(cm, ['Not Hot Lead', 'Hot Lead'])
        st.pyplot(fig)
    
    with col2:
        # ROC Curve
        st.subheader("📉 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("📊 Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance Ranking",
                    color='Importance', color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictions
    st.markdown("---")
    st.subheader("🔮 Make Predictions")
    
    # Predict on entire dataset
    X_all_scaled = scaler.transform(df[selected_features].fillna(df[selected_features].median()))
    df['Predicted_Hot_Lead'] = model.predict(X_all_scaled)
    df['Hot_Lead_Probability'] = model.predict_proba(X_all_scaled)[:, 1]
    
    # Show predictions
    display_cols = ['Industry', 'Employees', 'Interest_Level', 'Predicted_Hot_Lead', 'Hot_Lead_Probability']
    display_cols = [c for c in display_cols if c in df.columns]
    pred_df = df[display_cols].head(20)
    
    st.dataframe(fix_dtypes_for_arrow(pred_df), width="stretch")
    
    # Download predictions
    st.markdown(create_download_link(df, "classification_predictions.csv", 
                                    "📥 Download All Predictions"), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: CLUSTERING & SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def show_clustering(df):
    st.markdown("<h1 class='main-header'>👥 Customer Clustering & Segmentation</h1>", unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Segment customers into distinct groups based on behavior and characteristics")
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Clustering Settings")
    
    # Feature selection
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ['Cluster', 'Is_Hot_Lead', 
                                                                  'Predicted_Hot_Lead', 'Hot_Lead_Probability']]
    
    default_features = [f for f in ['Annual_Procurement_Spend_AED', 'Max_Monthly_WTP_AED',
                                    'Avg_Pain_Score', 'Avg_Feature_Value', 'Interest_Level',
                                    'Digital_Maturity_Score', 'Employees'] 
                       if f in numeric_features]
    
    if len(default_features) == 0:
        default_features = numeric_features[:7] if len(numeric_features) >= 7 else numeric_features
    
    selected_features = st.sidebar.multiselect(
        "Select Features for Clustering:",
        numeric_features,
        default=default_features
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features")
        return
    
    # Number of clusters
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)
    
    # Prepare data
    X = df[selected_features].fillna(df[selected_features].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow Method
    st.subheader("📉 Elbow Method - Optimal k")
    
    with st.spinner("🔄 Computing elbow plot..."):
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(11, len(df)//10))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Plot
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Elbow Plot', 'Silhouette Score'))
        
        fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers',
                                name='Inertia', line=dict(color='blue', width=2)),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers',
                                name='Silhouette', line=dict(color='green', width=2)),
                     row=1, col=2)
        
        fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Clusters (k)", row=1, col=2)
        fig.update_yaxes(title_text="Inertia (WCSS)", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Run K-Means with selected k
    st.markdown("---")
    st.subheader(f"🎯 K-Means Clustering (k={n_clusters})")
    
    with st.spinner("🔄 Running K-Means..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        sil_score = silhouette_score(X_scaled, df['Cluster'])
        st.success(f"✅ Clustering complete! Silhouette Score: {sil_score:.3f}")
    
    # Cluster distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cluster Distribution")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        fig = px.pie(values=cluster_counts.values, names=[f'Cluster {i}' for i in cluster_counts.index],
                    title="Customer Distribution by Cluster",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎨 Cluster Visualization (PCA)")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df['Cluster'].astype(str),
                        title=f"Clusters in 2D Space (PCA)",
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
                               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'},
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='black')))
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Profiles
    st.markdown("---")
    st.subheader("📋 Cluster Profiles")
    
    # Interactive persona naming
    persona_names = {}
    cols = st.columns(n_clusters)
    
    for i, col in enumerate(cols):
        with col:
            persona_names[i] = st.text_input(f"Cluster {i} Name:", 
                                            value=f"Segment {i}", 
                                            key=f"cluster_{i}")
    
    # Calculate profiles
    profile_features = [f for f in selected_features + ['Interest_Level'] if f in df.columns]
    
    cluster_profiles = df.groupby('Cluster')[profile_features].mean().round(2)
    cluster_profiles['Count'] = df['Cluster'].value_counts().sort_index()
    cluster_profiles['Percentage'] = (cluster_profiles['Count'] / len(df) * 100).round(1)
    
    # Rename clusters
    cluster_profiles.index = [persona_names.get(i, f"Cluster {i}") for i in cluster_profiles.index]
    
    st.dataframe(fix_dtypes_for_arrow(cluster_profiles), width="stretch")
    
    # Update dataframe with persona names
    df['Persona'] = df['Cluster'].map(persona_names)
    
    # Detailed profile comparison
    st.markdown("---")
    st.subheader("📊 Detailed Cluster Comparison")
    
    selected_metric = st.selectbox("Select metric to compare:", profile_features)
    
    fig = px.bar(cluster_profiles, y=selected_metric, 
                title=f"Comparison of {selected_metric} across Clusters",
                color=selected_metric, color_continuous_scale='Blues')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.markdown("---")
    st.markdown(create_download_link(df, "cluster_assignments.csv", 
                                    "📥 Download Cluster Assignments"), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: ASSOCIATION RULE MINING
# ═══════════════════════════════════════════════════════════════════════════

def show_association_rules(df):
    st.markdown("<h1 class='main-header'>🔗 Association Rule Mining</h1>", unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Discover patterns and associations between features, pain points, and behaviors")
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Apriori Settings")
    
    min_support = st.sidebar.slider("Minimum Support (%)", 1, 50, 6) / 100
    min_confidence = st.sidebar.slider("Minimum Confidence (%)", 30, 95, 65) / 100
    min_lift = st.sidebar.slider("Minimum Lift", 1.0, 3.0, 1.5, 0.1)
    top_n_rules = st.sidebar.slider("Top N Rules to Display", 5, 50, 10)
    
    # Select binary/categorical features
    binary_cols = []
    
    # Identify binary columns
    for col in df.columns:
        if df[col].nunique() <= 2 and set(df[col].dropna().unique()).issubset({0, 1, True, False, 0.0, 1.0}):
            binary_cols.append(col)
    
    # Convert high-value features to binary
    for col in ['Pain_Manual_RFQ', 'Pain_Vendor_Risk', 'Pain_Compliance']:
        if col in df.columns:
            df[f'High_{col}'] = (df[col] >= 4).astype(int)
            binary_cols.append(f'High_{col}')
    
    for col in ['Values_Automation', 'Values_Risk_Assessment', 'Values_Compliance']:
        if col in df.columns:
            df[f'High_{col}'] = (df[col] >= 4).astype(int)
            binary_cols.append(f'High_{col}')
    
    if len(binary_cols) < 3:
        st.warning("⚠️ Not enough binary features for association rule mining. Need at least 3 binary columns.")
        st.info("💡 Try uploading data with more binary columns or categorical variables.")
        return
    
    st.write(f"📊 Found {len(binary_cols)} binary features for analysis")
    
    # Prepare data for Apriori
    apriori_data = df[binary_cols].copy().fillna(0).astype(int)
    
    # Run Apriori
    with st.spinner("🔄 Mining association rules..."):
        try:
            frequent_itemsets = apriori(apriori_data, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                st.warning("⚠️ No frequent itemsets found. Try lowering the support threshold.")
                return
            
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            rules = rules[rules['lift'] >= min_lift]
            
            if len(rules) == 0:
                st.warning("⚠️ No rules found meeting criteria. Try lowering the thresholds.")
                return
            
            # Sort by lift
            rules = rules.sort_values('lift', ascending=False)
            
            st.success(f"✅ Found {len(rules)} association rules!")
            
        except Exception as e:
            st.error(f"❌ Error in association rule mining: {str(e)}")
            return
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Rules", len(rules))
    
    with col2:
        st.metric("📈 Avg Confidence", f"{rules['confidence'].mean():.2%}")
    
    with col3:
        st.metric("🚀 Avg Lift", f"{rules['lift'].mean():.2f}")
    
    # Top Rules
    st.markdown("---")
    st.subheader(f"🏆 Top {top_n_rules} Association Rules")
    
    top_rules = rules.head(top_n_rules).copy()
    
    # Format rules for display
    top_rules['antecedents_str'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    top_rules['consequents_str'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    display_rules = top_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].copy()
    display_rules.columns = ['IF (Antecedents)', 'THEN (Consequents)', 'Support', 'Confidence', 'Lift']
    display_rules['Support'] = display_rules['Support'].apply(lambda x: f"{x:.2%}")
    display_rules['Confidence'] = display_rules['Confidence'].apply(lambda x: f"{x:.2%}")
    display_rules['Lift'] = display_rules['Lift'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(fix_dtypes_for_arrow(display_rules), width="stretch")
    
    # Visualizations
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Support vs Confidence")
        fig = px.scatter(rules, x='support', y='confidence', size='lift',
                        color='lift', hover_data=['antecedents', 'consequents'],
                        title="Association Rules: Support vs Confidence",
                        color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Lift Distribution")
        fig = px.histogram(rules, x='lift', nbins=30,
                          title="Distribution of Lift Values",
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed rule interpretation
    st.markdown("---")
    st.subheader("🔍 Detailed Rule Interpretation")
    
    selected_rule_idx = st.selectbox("Select a rule to interpret:", 
                                     range(len(top_rules)),
                                     format_func=lambda x: f"Rule {x+1}")
    
    selected_rule = top_rules.iloc[selected_rule_idx]
    
    st.markdown(f"""
    ### Rule #{selected_rule_idx + 1}
    
    **IF:** {', '.join(list(selected_rule['antecedents']))}
    
    **THEN:** {', '.join(list(selected_rule['consequents']))}
    
    **Metrics:**
    - **Support:** {selected_rule['support']:.2%} ({int(selected_rule['support'] * len(df))} customers)
    - **Confidence:** {selected_rule['confidence']:.2%}
    - **Lift:** {selected_rule['lift']:.2f}x
    
    **Interpretation:**
    - {selected_rule['confidence']*100:.0f}% of customers who have {', '.join(list(selected_rule['antecedents']))} also have {', '.join(list(selected_rule['consequents']))}
    - This combination is {selected_rule['lift']:.1f}x more likely than random chance
    """)
    
    # Download rules
    st.markdown("---")
    rules_export = rules.copy()
    rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
    
    st.markdown(create_download_link(rules_export, "association_rules.csv", 
                                    "📥 Download All Rules"), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: REGRESSION & PRICING
# ═══════════════════════════════════════════════════════════════════════════

def show_regression(df):
    st.markdown("<h1 class='main-header'>📈 Regression: Predict Willingness to Pay</h1>", unsafe_allow_html=True)
    
    # Check if target exists
    if 'Max_Monthly_WTP_AED' not in df.columns:
        st.error("❌ Target variable 'Max_Monthly_WTP_AED' not found in dataset")
        st.info("💡 Please upload data with a 'Max_Monthly_WTP_AED' column or use synthetic data.")
        return
    
    st.info("🎯 **Objective:** Predict customer's maximum willingness to pay (WTP) for pricing optimization")
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Regression Settings")
    
    # Feature selection
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [f for f in numeric_features if f not in ['Max_Monthly_WTP_AED', 'Cluster', 
                                                                  'Predicted_Hot_Lead', 'Hot_Lead_Probability',
                                                                  'Predicted_WTP_AED', 'Prediction_Error_AED']]
    
    default_features = [f for f in ['Annual_Procurement_Spend_AED', 'Avg_Pain_Score', 
                                    'Interest_Level', 'Purchase_Urgency', 'Digital_Maturity_Score',
                                    'Avg_Feature_Value', 'Employees', 'Annual_Revenue_Million_AED'] 
                       if f in numeric_features]
    
    if len(default_features) == 0:
        default_features = numeric_features[:8] if len(numeric_features) >= 8 else numeric_features
    
    selected_features = st.sidebar.multiselect(
        "Select Features:",
        numeric_features,
        default=default_features
    )
    
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least 2 features")
        return
    
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 30) / 100
    
    model_choice = st.sidebar.selectbox(
        "Select Model:",
        ["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting"]
    )
    
    # Prepare data
    X = df[selected_features].fillna(df[selected_features].median())
    y = df['Max_Monthly_WTP_AED']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    with st.spinner("🔄 Training model..."):
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            alpha = st.sidebar.slider("Regularization (Alpha)", 0.1, 100.0, 10.0)
            model = Ridge(alpha=alpha, random_state=42)
        elif model_choice == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 50, 200, 100)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        else:
            n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, 100)
            model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Test R²", f"{test_r2:.3f}")
        st.caption(f"Train R²: {train_r2:.3f}")
    
    with col2:
        st.metric("📏 Test RMSE", f"AED {test_rmse:,.0f}")
        st.caption(f"Train: {train_rmse:,.0f}")
    
    with col3:
        st.metric("📊 Test MAE", f"AED {test_mae:,.0f}")
    
    with col4:
        st.metric("📈 Test MAPE", f"{test_mape:.1f}%")
    
    # Interpretation
    if test_r2 >= 0.80:
        st.success(f"✅ Excellent model! Explains {test_r2*100:.1f}% of variance in WTP")
    elif test_r2 >= 0.60:
        st.info(f"👍 Good model. Explains {test_r2*100:.1f}% of variance")
    else:
        st.warning(f"⚠️ Moderate performance. R² = {test_r2:.3f}")
    
    # Visualizations
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Actual vs Predicted")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                                name='Predictions',
                                marker=dict(size=8, opacity=0.6, color='blue')))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash', width=2)))
        fig.update_layout(
            xaxis_title='Actual WTP (AED)',
            yaxis_title='Predicted WTP (AED)',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Residuals Plot")
        
        residuals = y_test - y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                marker=dict(size=8, opacity=0.6, color='green')))
        fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        fig.update_layout(
            xaxis_title='Predicted WTP (AED)',
            yaxis_title='Residuals (AED)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("---")
    st.subheader("📊 Feature Importance")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="Feature Importance Ranking",
                    color='Importance', color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    elif hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': selected_features,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                    title="Feature Coefficients",
                    color='Coefficient', color_continuous_scale='RdBu')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictions
    st.markdown("---")
    st.subheader("🔮 Generate Predictions")
    
    X_all_scaled = scaler.transform(df[selected_features].fillna(df[selected_features].median()))
    df['Predicted_WTP_AED'] = model.predict(X_all_scaled)
    df['Prediction_Error_AED'] = test_mae
    
    # Show predictions
    pred_cols = ['Industry', 'Employees', 'Annual_Procurement_Spend_AED', 
                'Max_Monthly_WTP_AED', 'Predicted_WTP_AED']
    pred_cols = [c for c in pred_cols if c in df.columns]
    
    st.dataframe(fix_dtypes_for_arrow(df[pred_cols].head(20)), width="stretch")
    
    # Download predictions
    st.markdown(create_download_link(df, "wtp_predictions.csv", 
                                    "📥 Download All Predictions"), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6: DYNAMIC PRICING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def show_dynamic_pricing(df):
    st.markdown("<h1 class='main-header'>💰 Dynamic Pricing Engine</h1>", unsafe_allow_html=True)
    
    st.info("🎯 **Objective:** Generate personalized pricing recommendations based on predicted WTP and customer profile")
    
    # Check if predictions exist
    if 'Predicted_WTP_AED' not in df.columns:
        st.warning("⚠️ Please run Regression analysis first to generate WTP predictions")
        
        if st.button("🔮 Generate Predictions Now"):
            # Quick prediction using simple model
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [f for f in numeric_features if f not in ['Max_Monthly_WTP_AED', 'Cluster']]
            
            if 'Max_Monthly_WTP_AED' in df.columns and len(numeric_features) >= 3:
                X = df[numeric_features[:5]].fillna(df[numeric_features[:5]].median())
                y = df['Max_Monthly_WTP_AED']
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_scaled, y)
                
                df['Predicted_WTP_AED'] = model.predict(X_scaled)
                st.success("✅ Predictions generated! Refresh the page.")
            else:
                st.error("❌ Cannot generate predictions. Missing required columns.")
        return
    
    # Pricing tiers configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Pricing Tiers")
    
    tier_names = ['Starter', 'Growth', 'Professional', 'Enterprise', 'Enterprise Plus']
    tier_prices = {}
    
    default_prices = {
        'Starter': 999,
        'Growth': 2499,
        'Professional': 4999,
        'Enterprise': 7999,
        'Enterprise Plus': 12999
    }
    
    for tier in tier_names:
        tier_prices[tier] = st.sidebar.number_input(
            f"{tier} Price (AED/month):",
            min_value=500,
            max_value=20000,
            value=default_prices[tier],
            step=100
        )
    
    # Discount settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎁 Discount Strategy")
    
    hot_lead_discount = st.sidebar.slider("Hot Lead Discount (%)", 0, 30, 15)
    annual_discount = st.sidebar.slider("Annual Payment Discount (%)", 0, 30, 20)
    
    # Assign tiers
    def assign_tier(wtp):
        if wtp < 1500:
            return 'Starter'
        elif wtp < 3500:
            return 'Growth'
        elif wtp < 6000:
            return 'Professional'
        elif wtp < 10000:
            return 'Enterprise'
        else:
            return 'Enterprise Plus'
    
    df['Recommended_Tier'] = df['Predicted_WTP_AED'].apply(assign_tier)
    df['List_Price'] = df['Recommended_Tier'].map(tier_prices)
    
    # Apply discounts
    df['Discount_Percent'] = 0
    if 'Is_Hot_Lead' in df.columns:
        df.loc[df['Is_Hot_Lead'] == 1, 'Discount_Percent'] = hot_lead_discount
    elif 'Interest_Level' in df.columns:
        df.loc[df['Interest_Level'] >= 4, 'Discount_Percent'] = 10
    
    df['Recommended_Price'] = (df['List_Price'] * (1 - df['Discount_Percent']/100)).astype(int)
    df['Price_to_WTP_Ratio'] = df['Recommended_Price'] / df['Predicted_WTP_AED']
    
    # Calculate annual pricing
    df['Annual_Price_Full'] = df['Recommended_Price'] * 12
    df['Annual_Price_Discounted'] = (df['Annual_Price_Full'] * (1 - annual_discount/100)).astype(int)
    df['Annual_Savings'] = df['Annual_Price_Full'] - df['Annual_Price_Discounted']
    
    # Overview metrics
    st.subheader("📊 Pricing Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_monthly = df['Recommended_Price'].sum()
        st.metric("💰 Total Monthly Revenue", f"AED {total_monthly:,.0f}")
    
    with col2:
        total_annual = total_monthly * 12
        st.metric("📅 Total Annual Revenue", f"AED {total_annual:,.0f}")
    
    with col3:
        avg_price = df['Recommended_Price'].mean()
        st.metric("📊 Avg Price/Customer", f"AED {avg_price:,.0f}")
    
    with col4:
        avg_ratio = df['Price_to_WTP_Ratio'].mean()
        st.metric("🎯 Avg Price/WTP Ratio", f"{avg_ratio:.2f}")
    
    # Tier distribution
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Customer Distribution by Tier")
        
        tier_counts = df['Recommended_Tier'].value_counts()
        tier_counts = tier_counts.reindex(tier_names, fill_value=0)
        
        fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                    title="Customers by Pricing Tier",
                    color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Revenue by Tier")
        
        tier_revenue = df.groupby('Recommended_Tier')['Recommended_Price'].sum()
        tier_revenue = tier_revenue.reindex(tier_names, fill_value=0)
        
        fig = px.bar(x=tier_revenue.index, y=tier_revenue.values,
                    title="Monthly Revenue by Tier",
                    color=tier_revenue.values,
                    color_continuous_scale='Greens',
                    labels={'x': 'Tier', 'y': 'Revenue (AED)'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed tier analysis
    st.markdown("---")
    st.subheader("📋 Detailed Tier Analysis")
    
    tier_analysis = df.groupby('Recommended_Tier').agg({
        'Recommended_Price': ['count', 'mean', 'sum'],
        'Predicted_WTP_AED': 'mean',
        'Price_to_WTP_Ratio': 'mean',
        'Discount_Percent': 'mean'
    }).round(0)
    
    tier_analysis.columns = ['Customer_Count', 'Avg_Price', 'Total_Revenue', 
                            'Avg_WTP', 'Avg_Price_WTP_Ratio', 'Avg_Discount']
    tier_analysis = tier_analysis.reindex(tier_names, fill_value=0)
    
    st.dataframe(fix_dtypes_for_arrow(tier_analysis), width="stretch")
    
    # Price optimization visualization
    st.markdown("---")
    st.subheader("🎯 Price Optimization Analysis")
    
    fig = px.scatter(df, x='Predicted_WTP_AED', y='Recommended_Price',
                    color='Recommended_Tier',
                    size='Annual_Procurement_Spend_AED' if 'Annual_Procurement_Spend_AED' in df.columns else None,
                    hover_data=['Industry'] if 'Industry' in df.columns else None,
                    title="Recommended Price vs Predicted WTP",
                    color_discrete_sequence=px.colors.qualitative.Set3)
    
    # Add perfect pricing line
    max_val = max(df['Predicted_WTP_AED'].max(), df['Recommended_Price'].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                            mode='lines', name='Price = WTP',
                            line=dict(color='red', dash='dash', width=2)))
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Customer prioritization
    st.markdown("---")
    st.subheader("🎯 Customer Prioritization")
    
    # Calculate value score
    df['Customer_Value_Score'] = (
        df['Predicted_WTP_AED'] / 100 +
        df.get('Interest_Level', 3) * 10 +
        df.get('Is_Hot_Lead', 0) * 50
    )
    
    # Segment customers
    def priority_segment(row):
        if row.get('Is_Hot_Lead', 0) == 1 and row['Predicted_WTP_AED'] >= 5000:
            return 'A - High Value Hot Leads'
        elif row.get('Is_Hot_Lead', 0) == 1:
            return 'B - Hot Leads'
        elif row.get('Interest_Level', 3) >= 4 and row['Predicted_WTP_AED'] >= 3000:
            return 'C - High Intent'
        elif row['Predicted_WTP_AED'] >= 5000:
            return 'D - High Value'
        else:
            return 'E - Standard'
    
    df['Priority_Segment'] = df.apply(priority_segment, axis=1)
    
    segment_summary = df.groupby('Priority_Segment').agg({
        'Customer_Value_Score': 'mean',
        'Recommended_Price': ['count', 'sum']
    }).round(0)
    
    segment_summary.columns = ['Avg_Value_Score', 'Customer_Count', 'Total_Revenue']
    
    st.dataframe(fix_dtypes_for_arrow(segment_summary), width="stretch")
    
    # Sample recommendations
    st.markdown("---")
    st.subheader("📋 Sample Pricing Recommendations")
    
    display_cols = ['Industry', 'Employees', 'Predicted_WTP_AED', 'Recommended_Tier',
                   'List_Price', 'Discount_Percent', 'Recommended_Price', 
                   'Annual_Price_Discounted', 'Priority_Segment']
    display_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(fix_dtypes_for_arrow(df[display_cols].head(20)), width="stretch")
    
    # Download full recommendations
    st.markdown("---")
    st.markdown(create_download_link(df, "dynamic_pricing_recommendations.csv", 
                                    "📥 Download Complete Pricing Recommendations"), 
               unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# RUN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
