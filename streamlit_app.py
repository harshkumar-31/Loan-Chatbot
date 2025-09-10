import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from nbfc_chatbot_system import NBFCLoanCollectionSystem

# Page configuration
st.set_page_config(
    page_title="NBFC Collection Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f4e79;
        padding-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .strategy-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .persona-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'nbfc_system' not in st.session_state:
    st.session_state.nbfc_system = NBFCLoanCollectionSystem()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'customer_data' not in st.session_state:
    st.session_state.customer_data = {}

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Main header
st.markdown('<h1 class="main-header">🏦 NBFC AI Collection Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Intelligent loan collection management with AI-powered persona detection and risk assessment</p>', unsafe_allow_html=True)

# Sidebar for customer profile
st.sidebar.header("👤 Customer Profile")

with st.sidebar:
    st.subheader("Personal Information")
    customer_id = st.text_input("Customer ID", value="CUST001")
    age = st.slider("Age", 18, 80, 35)
    income = st.number_input("Annual Income (₹)", min_value=100000, max_value=10000000, value=800000, step=50000)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Student", "Unemployed"])
    
    st.subheader("Loan Details")
    loan_amount = st.number_input("Loan Amount (₹)", min_value=50000, max_value=2000000, value=500000, step=10000)
    tenure_months = st.slider("Tenure (Months)", 6, 60, 36)
    interest_rate = st.slider("Interest Rate (%)", 6.0, 18.0, 12.5, 0.1)
    loan_type = st.selectbox("Loan Type", ["Personal", "Auto", "Home", "Education", "Business"])
    
    st.subheader("Payment History")
    missed_payments = st.slider("Missed Payments", 0, 10, 2)
    delays_days = st.slider("Total Delay Days", 0, 200, 45)
    partial_payments = st.slider("Partial Payments", 0, 10, 1)
    
    st.subheader("Interaction Data")
    interaction_attempts = st.slider("Interaction Attempts", 0, 15, 3)
    sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, -0.3, 0.1)
    response_time_hours = st.slider("Avg Response Time (Hours)", 0.0, 72.0, 24.0, 1.0)
    
    st.subheader("Engagement Metrics")
    app_usage_freq = st.slider("App Usage Frequency", 0.0, 1.0, 0.6, 0.1)
    website_visits = st.slider("Website Visits", 0, 50, 15)
    complaints = st.slider("Complaints", 0, 5, 1)

# Update customer data
st.session_state.customer_data = {
    'CustomerID': customer_id,
    'Age': age,
    'Income': income,
    'Location': location,
    'EmploymentStatus': employment,
    'LoanAmount': loan_amount,
    'TenureMonths': tenure_months,
    'InterestRate': interest_rate,
    'LoanType': loan_type,
    'MissedPayments': missed_payments,
    'DelaysDays': delays_days,
    'PartialPayments': partial_payments,
    'InteractionAttempts': interaction_attempts,
    'SentimentScore': sentiment_score,
    'ResponseTimeHours': response_time_hours,
    'AppUsageFrequency': app_usage_freq,
    'WebsiteVisits': website_visits,
    'Complaints': complaints
}

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Interface", "📊 Risk Analytics", "🎯 Strategy Dashboard", "📈 Analytics"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Customer Conversation")
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>Customer:</strong> {chat['user_message']}
                </div>
                """, unsafe_allow_html=True)
                
                # Bot response
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>AI Assistant:</strong> {chat['bot_response']}<br>
                    <small><em>Persona: {chat.get('persona', 'Unknown')} | Risk: {chat.get('risk_level', 'Unknown')}</em></small>
                </div>
                """, unsafe_allow_html=True)
        
        # Input for new message
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Type your message here...", height=100, placeholder="Enter customer message...")
            submitted = st.form_submit_button("Send Message", use_container_width=True)
            
            if submitted and user_input.strip():
                # Generate response
                response_data = st.session_state.nbfc_system.generate_response(user_input, st.session_state.customer_data)
                
                # Add to chat history
                chat_entry = {
                    'user_message': user_input,
                    'bot_response': response_data['response'],
                    'persona': response_data['persona'],
                    'risk_score': response_data['risk_score'],
                    'risk_level': response_data['risk_level'],
                    'strategy': response_data['strategy'],
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.chat_history.append(chat_entry)
                st.rerun()
    
    with col2:
        st.subheader("📊 Real-time Insights")
        
        if st.session_state.chat_history:
            latest_chat = st.session_state.chat_history[-1]
            
            # Risk Score Gauge
            risk_score = latest_chat['risk_score']
            
            # Create risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Current Persona
            st.markdown(f"""
            <div class="persona-card">
                <h4>Current Persona</h4>
                <h2>{latest_chat['persona'].title()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Strategy Summary
            strategy = latest_chat['strategy']
            st.markdown(f"""
            <div class="strategy-card">
                <h4>Recommended Strategy</h4>
                <p><strong>Approach:</strong> {strategy['approach']}</p>
                <p><strong>Tone:</strong> {strategy['tone'].title()}</p>
                <p><strong>Timeline:</strong> {strategy['timeline']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.info("Start a conversation to see real-time insights")

with tab2:
    st.subheader("📊 Risk Analytics Dashboard")
    
    # Model training section
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained. Upload dataset to train the model.")
        
        uploaded_file = st.file_uploader("Upload Training Dataset (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                if st.button("Train Model", type="primary"):
                    with st.spinner("Training model..."):
                        accuracy, roc_auc = st.session_state.nbfc_system.train_model(df)
                        st.session_state.model_trained = True
                        st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}")
                        st.rerun()
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
    
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Current customer risk
            if st.session_state.customer_data:
                try:
                    risk_prob = st.session_state.nbfc_system.predict_risk(st.session_state.customer_data)
                    risk_level = st.session_state.nbfc_system.get_risk_level(risk_prob)
                    
                    st.markdown(f"""
                    <div class="risk-card">
                        <h3>Current Customer Risk</h3>
                        <h1>{risk_prob:.1%}</h1>
                        <p>Risk Level: {risk_level.replace('_', ' ').title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    st.error("Error calculating risk score")
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Payment Behavior Score</h4>
                <h2>7.2/10</h2>
                <p>Based on payment history</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>Engagement Score</h4>
                <h2>6.8/10</h2>
                <p>App usage & website visits</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk factors breakdown
        st.subheader("🎯 Risk Factors Analysis")
        
        if st.session_state.customer_data:
            risk_factors = {
                'Missed Payments': st.session_state.customer_data['MissedPayments'] * 10,
                'Delay Days': min(st.session_state.customer_data['DelaysDays'] / 2, 100),
                'Complaints': st.session_state.customer_data['Complaints'] * 20,
                'Sentiment Score': max(0, (1 - st.session_state.customer_data['SentimentScore']) * 50),
                'LTV Ratio': min((st.session_state.customer_data['LoanAmount'] / st.session_state.customer_data['Income']) * 100, 100),
                'Response Time': min(st.session_state.customer_data['ResponseTimeHours'], 100)
            }
            
            fig_factors = px.bar(
                x=list(risk_factors.values()),
                y=list(risk_factors.keys()),
                orientation='h',
                title="Risk Factors Contribution",
                color=list(risk_factors.values()),
                color_continuous_scale='Reds'
            )
            fig_factors.update_layout(height=400)
            st.plotly_chart(fig_factors, use_container_width=True)

with tab3:
    st.subheader("🎯 Collection Strategy Dashboard")
    
    if st.session_state.chat_history:
        latest_chat = st.session_state.chat_history[-1]
        persona = latest_chat['persona']
        strategy = latest_chat['strategy']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📋 Current Strategy Details")
            
            st.markdown(f"""
            **Customer Persona:** {persona.title()}  
            **Risk Level:** {latest_chat['risk_level'].replace('_', ' ').title()}  
            **Recommended Approach:** {strategy['approach']}  
            **Communication Tone:** {strategy['tone'].title()}  
            **Timeline:** {strategy['timeline']}  
            **Escalation Path:** {strategy['escalation']}
            """)
            
            st.markdown("### 🛠️ Key Tactics")
            for i, tactic in enumerate(strategy['tactics'], 1):
                st.markdown(f"{i}. {tactic}")
            
            # Strategy effectiveness simulation
            st.markdown("### 📈 Strategy Effectiveness Prediction")
            
            effectiveness_data = {
                'Approach': ['Current Strategy', 'Alternative 1', 'Alternative 2'],
                'Success Rate': [85, 72, 68],
                'Recovery Amount': [450000, 380000, 340000],
                'Time to Resolution': [12, 18, 25]
            }
            
            effectiveness_df = pd.DataFrame(effectiveness_data)
            st.dataframe(effectiveness_df, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Strategy Metrics")
            
            # Success rate gauge
            fig_success = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 85,  # Simulated success rate
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Success Rate %"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig_success.update_layout(height=250)
            st.plotly_chart(fig_success, use_container_width=True)
            
            # Timeline visualization
            timeline_data = {
                'Day': list(range(1, 15)),
                'Action': ['Initial Contact', 'Follow-up', 'Strategy Review', 'Escalation'] * 3 + ['Resolution', 'Close', 'Post-Resolution'],
                'Status': ['Completed'] * 7 + ['Planned'] * 7
            }
            
            timeline_df = pd.DataFrame(timeline_data)
            fig_timeline = px.timeline(
                timeline_df[:7], 
                x_start='Day', 
                x_end='Day',
                y='Action',
                title="Collection Timeline"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    else:
        st.info("Start a conversation to see strategy recommendations")

with tab4:
    st.subheader("📈 Analytics & Reporting")
    
    # Get analytics from the system
    analytics = st.session_state.nbfc_system.get_analytics_summary()
    
    if analytics.get('total_conversations', 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Conversations", analytics['total_conversations'])
        
        with col2:
            st.metric("Average Risk Score", f"{analytics.get('average_risk_score', 0):.2%}")
        
        with col3:
            st.metric("Most Common Persona", analytics.get('most_common_persona', 'N/A').title())
        
        with col4:
            st.metric("Active Customers", len(set([chat.get('customer_id', 'Unknown') for chat in st.session_state.chat_history])))
        
        # Persona distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if analytics.get('persona_distribution'):
                fig_persona = px.pie(
                    values=list(analytics['persona_distribution'].values()),
                    names=list(analytics['persona_distribution'].keys()),
                    title="Persona Distribution"
                )
                st.plotly_chart(fig_persona, use_container_width=True)
        
        with col2:
            if analytics.get('risk_level_distribution'):
                fig_risk = px.bar(
                    x=list(analytics['risk_level_distribution'].keys()),
                    y=list(analytics['risk_level_distribution'].values()),
                    title="Risk Level Distribution",
                    color=list(analytics['risk_level_distribution'].values()),
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
        
        # Conversation timeline
        if st.session_state.chat_history:
            conversation_df = pd.DataFrame(st.session_state.chat_history)
            conversation_df['timestamp'] = pd.to_datetime(conversation_df['timestamp'])
            conversation_df['hour'] = conversation_df['timestamp'].dt.hour
            
            hourly_activity = conversation_df.groupby('hour').size().reset_index(name='conversations')
            
            fig_timeline = px.line(
                hourly_activity,
                x='hour',
                y='conversations',
                title="Conversation Activity by Hour",
                markers=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Export functionality
        st.subheader("📥 Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Conversation History"):
                filename = st.session_state.nbfc_system.export_conversation_history()
                st.success(f"Exported to: {filename}")
        
        with col2:
            if st.button("Download Analytics Report"):
                analytics_df = pd.DataFrame([analytics])
                csv = analytics_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("No conversation data available. Start chatting to see analytics.")
        
        # Sample data visualization
        st.subheader("📊 Sample Analytics")
        
        sample_data = {
            'Persona': ['Cooperative', 'Evasive', 'Aggressive', 'Confused', 'Financial Distress'],
            'Count': [45, 28, 12, 15, 20],
            'Avg Risk Score': [0.3, 0.6, 0.8, 0.4, 0.7],
            'Success Rate': [85, 65, 45, 75, 60]
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sample = px.bar(
                sample_df,
                x='Persona',
                y='Count',
                title="Sample Persona Distribution",
                color='Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_sample, use_container_width=True)
        
        with col2:
            fig_success = px.scatter(
                sample_df,
                x='Avg Risk Score',
                y='Success Rate',
                size='Count',
                color='Persona',
                title="Risk Score vs Success Rate"
            )
            st.plotly_chart(fig_success, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🏦 NBFC AI Collection Assistant | Powered by Machine Learning & Natural Language Processing</p>
    <p><small>© 2024 - Intelligent Loan Collection Management System</small></p>
</div>
""", unsafe_allow_html=True)