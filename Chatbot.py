# Enhanced NBFC Loan Collection Chatbot - Professional Version
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure Streamlit page
st.set_page_config(
    page_title="NBFC AI Collection Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f5f9fc 0%, #e8f4f8 100%);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main Header */
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.1;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Chat Container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.1);
        margin-bottom: 1rem;
        min-height: 400px;
    }
    
    /* Message Styling */
    .customer-message {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        animation: slideInRight 0.3s ease-out;
        position: relative;
    }
    
    .agent-message {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        color: #1e293b;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 75%;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        animation: slideInLeft 0.3s ease-out;
        border-left: 4px solid #3b82f6;
    }
    
    .message-meta {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e40af;
        margin: 0.5rem 0;
    }
    
    /* Risk Badges */
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(239, 68, 68, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(245, 158, 11, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
    }
    
    /* Persona Badges */
    .persona-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .persona-cooperative { background: #dbeafe; color: #1e40af; }
    .persona-aggressive { background: #fee2e2; color: #dc2626; }
    .persona-confused { background: #fef3c7; color: #d97706; }
    .persona-evasive { background: #f3e8ff; color: #7c3aed; }
    .persona-neutral { background: #f1f5f9; color: #64748b; }
    
    /* Strategy Card */
    .strategy-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #3b82f6;
    }
    
    .strategy-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    /* Animations */
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in { animation: fadeIn 0.5s ease-out; }
    
    /* Welcome Message */
    .welcome-message {
        text-align: center;
        padding: 3rem 2rem;
        color: #64748b;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 15px;
        border: 2px dashed #cbd5e1;
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 1rem;
    }
    
    /* Instructions Panel */
    .instructions-panel {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    .instruction-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .instruction-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
    }
    
    .instruction-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .instruction-title {
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'customer_persona' not in st.session_state:
    st.session_state.customer_persona = "Neutral"
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()

class AdvancedLoanCollectionChatbot:
    def __init__(self):
        self.conversation_history = []
        self.persona_weights = {
            'Cooperative': ['understand', 'help', 'try', 'want', 'willing', 'cooperate', 'sorry', 'apologize', 'work together', 'resolve'],
            'Aggressive': ['angry', 'frustrated', 'upset', 'terrible', 'awful', 'hate', 'worst', 'mad', 'ridiculous', 'unfair'],
            'Confused': ['confused', "don't understand", 'explain', 'what', 'how', 'unclear', 'lost', 'complicated', 'clarify'],
            'Evasive': ['busy', 'later', 'not now', "can't talk", 'call back', 'away', 'traveling', 'unavailable', 'another time'],
            'Neutral': []
        }
        
    def detect_persona_from_message(self, message):
        """Advanced persona detection using weighted keywords"""
        message_lower = message.lower()
        persona_scores = {}
        
        for persona, keywords in self.persona_weights.items():
            score = sum(2 if keyword in message_lower else 0 for keyword in keywords)
            # Add partial matching bonus
            score += sum(0.5 for keyword in keywords if any(word in keyword for word in message_lower.split()))
            persona_scores[persona] = score
        
        if all(score == 0 for score in persona_scores.values()):
            return "Neutral"
        
        return max(persona_scores, key=persona_scores.get)
    
    def calculate_risk_score(self, customer_data):
        """Enhanced risk calculation with multiple factors"""
        risk_score = 0.3  # Base risk
        
        # Age factor (U-shaped curve)
        age = customer_data['age']
        if age < 25 or age > 65:
            risk_score += 0.15
        elif 35 <= age <= 45:
            risk_score -= 0.05
        
        # Income factor (non-linear)
        income = customer_data['income']
        if income < 300000:
            risk_score += 0.25
        elif income < 500000:
            risk_score += 0.15
        elif income > 2000000:
            risk_score -= 0.10
        
        # Sentiment impact
        sentiment = customer_data['sentiment']
        risk_score += (0 - sentiment) * 0.2  # Negative sentiment increases risk
        
        # Payment history (exponential impact)
        missed_payments = customer_data['missed_payments']
        risk_score += missed_payments * 0.12 + (missed_payments ** 1.5) * 0.03
        
        # Complaints (compound effect)
        complaints = customer_data['complaints']
        risk_score += complaints * 0.08 + (complaints ** 1.2) * 0.02
        
        # Engagement (inverse relationship)
        engagement = customer_data['engagement']
        risk_score += (0.7 - engagement) * 0.15
        
        # Normalize to [0, 1] range
        risk_probability = max(0.05, min(0.95, risk_score))
        
        if risk_probability >= 0.75:
            return "High", risk_probability
        elif risk_probability >= 0.35:
            return "Medium", risk_probability
        else:
            return "Low", risk_probability
    
    def generate_contextual_response(self, persona, context, risk_level, customer_data=None):
        """Generate sophisticated contextual responses"""
        
        responses = {
            "Cooperative": {
                "greeting": "Thank you for taking my call today. I genuinely appreciate your time and want to work together to find the best solution for your situation.",
                "payment": "I understand that managing finances can be challenging, especially in today's economic climate. Let's explore some flexible payment options that could work better for your current circumstances.",
                "follow_up": "I'm grateful for your cooperative spirit. Let's create a realistic payment plan that fits comfortably within your budget while addressing your account needs.",
                "closing": "Thank you so much for your patience and willingness to work with us today. I'll ensure you receive written confirmation of our agreement, and please don't hesitate to reach out if you need any adjustments.",
                "complaint": "I hear your concerns and want to address them properly. Your feedback helps us improve our service. Let's work together to resolve this situation."
            },
            "Aggressive": {
                "greeting": "Good day. I understand this situation may be causing you frustration, and I want to address your concerns in a professional and respectful manner.",
                "payment": "I can hear your frustration, and that's completely understandable. Let's focus on finding a practical solution that works for both parties. What payment arrangement would feel most manageable for you?",
                "follow_up": "I appreciate you taking the time to discuss this with me despite your concerns. Let's establish a clear, mutually acceptable agreement that addresses your situation.",
                "closing": "Thank you for your directness today. I'll document our agreement carefully and ensure you receive all details in writing within 24 hours.",
                "complaint": "I understand your frustration, and your concerns are valid. Let me see how we can address these issues while finding a path forward for your account."
            },
            "Evasive": {
                "greeting": "Hello, I know you have a busy schedule, so I'll be respectful of your time. This is regarding your loan account, and I'd like to find a convenient time to discuss your options.",
                "payment": "I completely understand that you have many priorities competing for your attention. However, it would be beneficial for us to address your account status. When would be the most convenient time this week for a brief discussion about payment arrangements?",
                "follow_up": "I respect your time constraints. Could we schedule a specific 10-15 minute window that works better for you? I want to make this as convenient as possible.",
                "closing": "I'll follow up at the time we've agreed upon. I'll also send you my direct contact information so you can reach me when it's convenient for you.",
                "complaint": "I understand you're busy, and I don't want to add to your stress. Let's find a quick, efficient way to address both your concerns and your account status."
            },
            "Confused": {
                "greeting": "Hello! I'm calling to help clarify your loan account situation. I'll explain everything step by step in clear terms and answer any questions you might have.",
                "payment": "Let me break down your account details in simple, straightforward terms. Your payment of ₹[amount] was due on [date]. I'd be happy to explain the different payment options available and help you choose what works best.",
                "follow_up": "Let me summarize our conversation to ensure everything is crystal clear. Do you have any questions about the payment plan we've outlined? I want to make sure you're completely comfortable with the arrangement.",
                "closing": "I'll send you a detailed written summary of our conversation and your payment options via email and text. Please feel free to call me directly if you need any clarification.",
                "complaint": "I understand this situation might seem overwhelming. Let me explain everything clearly and address your concerns step by step."
            },
            "Neutral": {
                "greeting": "Good day. I'm calling regarding your loan account to discuss your current payment status and explore available options that might work for your situation.",
                "payment": "Your payment of ₹[amount] was due on [date]. I'd like to discuss the most practical way to bring your account current. We have several options available.",
                "follow_up": "Based on our discussion, here are the payment options that might work for your situation. Which approach feels most manageable for you?",
                "closing": "Thank you for your time today. You should receive confirmation of our arrangement within 24 hours, along with all relevant details.",
                "complaint": "I understand your concerns about this situation. Let me address your specific issues while we work on resolving your account status."
            }
        }
        
        base_response = responses[persona].get(context, responses[persona]["greeting"])
        
        # Risk-based modifications
        if risk_level == "High":
            if context == "payment":
                base_response += " Given the current status of your account, I'd like to prioritize finding a solution that works for both of us as quickly as possible."
            elif context == "closing":
                base_response += " I'll also schedule a follow-up call to ensure everything is progressing smoothly."
        elif risk_level == "Low":
            base_response = base_response.replace("it's important", "it would be helpful")
            base_response = base_response.replace("need to", "could")
        
        return base_response
    
    def get_collection_strategy(self, persona, risk_level, customer_data=None):
        """Comprehensive collection strategy recommendations"""
        
        strategies = {
            "High": {
                "Cooperative": {
                    "approach": "Empathetic Partnership Approach",
                    "tactics": [
                        "Schedule immediate personal call with senior collection specialist",
                        "Offer multiple payment plan options with immediate start dates",
                        "Provide hardship program information if applicable",
                        "Set up automatic payment arrangements",
                        "Assign dedicated account manager for ongoing support"
                    ],
                    "timeline": "Immediate action within 24 hours"
                },
                "Evasive": {
                    "approach": "Multi-Channel Persistence Strategy",
                    "tactics": [
                        "Implement systematic contact schedule across all channels",
                        "Send formal written communication via registered mail",
                        "Schedule specific callback appointments",
                        "Prepare preliminary legal notice documentation",
                        "Engage third-party mediation services"
                    ],
                    "timeline": "Escalated contact over 7-10 days"
                },
                "Aggressive": {
                    "approach": "Professional Boundary Management",
                    "tactics": [
                        "Assign experienced specialist trained in de-escalation",
                        "Document all interactions meticulously",
                        "Implement immediate escalation protocols",
                        "Offer structured payment alternatives",
                        "Prepare for potential legal proceedings"
                    ],
                    "timeline": "Immediate specialized intervention"
                },
                "Confused": {
                    "approach": "Educational Support Strategy",
                    "tactics": [
                        "Provide comprehensive account education session",
                        "Create simplified payment plan documentation",
                        "Assign dedicated customer service support",
                        "Offer financial counseling resources",
                        "Set up reminder systems with clear instructions"
                    ],
                    "timeline": "Extended support over 2-3 weeks"
                },
                "Neutral": {
                    "approach": "Standard Intensive Follow-up",
                    "tactics": [
                        "Implement regular follow-up schedule",
                        "Offer standard payment arrangement options",
                        "Monitor account closely for changes",
                        "Prepare escalation procedures",
                        "Document all interactions thoroughly"
                    ],
                    "timeline": "Weekly contact for 4 weeks"
                }
            },
            "Medium": {
                "Cooperative": {
                    "approach": "Collaborative Resolution",
                    "tactics": [
                        "Schedule friendly reminder calls",
                        "Offer flexible payment arrangements",
                        "Maintain positive customer relationship",
                        "Provide self-service options",
                        "Monitor for early success indicators"
                    ],
                    "timeline": "Bi-weekly follow-up"
                },
                "Evasive": {
                    "approach": "Incentivized Engagement",
                    "tactics": [
                        "Send email and SMS reminder campaigns",
                        "Offer early payment incentives",
                        "Schedule specific callback appointments",
                        "Provide multiple contact channels",
                        "Use behavioral nudge techniques"
                    ],
                    "timeline": "Multiple touchpoints over 2 weeks"
                },
                "Aggressive": {
                    "approach": "Firm But Fair Communication",
                    "tactics": [
                        "Use professional but clear communication",
                        "Document consequences of non-payment",
                        "Offer reasonable payment alternatives",
                        "Maintain respectful but firm boundaries",
                        "Monitor for escalation triggers"
                    ],
                    "timeline": "Structured contact every 3-5 days"
                },
                "Confused": {
                    "approach": "Guided Support Process",
                    "tactics": [
                        "Provide clear, simple reminders",
                        "Offer step-by-step payment guidance",
                        "Share educational resources",
                        "Provide customer support contact information",
                        "Use visual aids and simple language"
                    ],
                    "timeline": "Regular check-ins weekly"
                },
                "Neutral": {
                    "approach": "Standard Systematic Follow-up",
                    "tactics": [
                        "Implement standard reminder protocols",
                        "Offer typical payment arrangements",
                        "Monitor account activity regularly",
                        "Provide self-service options",
                        "Maintain consistent communication"
                    ],
                    "timeline": "Standard 10-day cycle"
                }
            },
            "Low": {
                "Cooperative": {
                    "approach": "Relationship Maintenance",
                    "tactics": [
                        "Send gentle, friendly reminders",
                        "Focus on preserving customer relationship",
                        "Offer convenient payment methods",
                        "Provide loyalty program information",
                        "Use positive reinforcement techniques"
                    ],
                    "timeline": "Monthly courtesy contacts"
                },
                "Evasive": {
                    "approach": "Minimal Intervention",
                    "tactics": [
                        "Use automated reminder systems",
                        "Respect preferred communication channels",
                        "Provide convenient self-service options",
                        "Minimal personal intervention",
                        "Monitor for voluntary payments"
                    ],
                    "timeline": "Automated monthly reminders"
                },
                "Aggressive": {
                    "approach": "Professional Courtesy",
                    "tactics": [
                        "Send professional courtesy reminders",
                        "Avoid escalation unless absolutely necessary",
                        "Provide clear payment instructions",
                        "Maintain respectful tone throughout",
                        "Focus on service recovery"
                    ],
                    "timeline": "Minimal contact, monthly check-ins"
                },
                "Confused": {
                    "approach": "Simple Guidance",
                    "tactics": [
                        "Use automated reminders with clear instructions",
                        "Provide simple payment guides",
                        "Offer easy-to-understand contact information",
                        "Use visual payment reminders",
                        "Minimize complexity in all communications"
                    ],
                    "timeline": "Simple monthly reminders"
                },
                "Neutral": {
                    "approach": "Standard Automated Process",
                    "tactics": [
                        "Implement standard automated reminder system",
                        "Provide routine payment options",
                        "Monitor for changes in payment patterns",
                        "Maintain standard service levels",
                        "Use cost-effective communication channels"
                    ],
                    "timeline": "Standard automated cycle"
                }
            }
        }
        
        return strategies[risk_level][persona]

# Main Application Header
st.markdown("""
<div class="header-container">
    <div class="main-title">🏦 NBFC AI Collection Assistant</div>
    <div class="main-subtitle">Professional Loan Collection Management System</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <h3 style="color: #1e40af; margin-bottom: 1rem;">👤 Customer Profile</h3>
        <p style="color: #64748b; margin: 0;">Configure customer parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Customer Profile Inputs
    col1, col2 = st.columns(2)
    with col1:
        customer_age = st.number_input("🎂 Age", min_value=18, max_value=80, value=35)
        customer_sentiment = st.slider("😊 Sentiment", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
        complaints = st.number_input("📞 Complaints", min_value=0, max_value=10, value=1)
    
    with col2:
        missed_payments = st.number_input("⚠️ Missed Payments", min_value=0, max_value=12, value=2)
        engagement = st.slider("📱 Engagement", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        
    customer_income = st.number_input("💰 Annual Income (₹)", min_value=100000, max_value=5000000, value=800000, step=50000)
    
    # Quick Persona Test Buttons
    st.markdown("### 🎭 Quick Persona Tests")
    test_messages = {
        "😠 Aggressive": "I'm really angry about these constant calls!",
        "😊 Cooperative": "I understand and want to help resolve this",
        "😕 Confused": "I don't understand why I owe so much",
        "😐 Evasive": "I'm busy right now, call me later"
    }
    
    for emoji_persona, message in test_messages.items():
        if st.button(emoji_persona, use_container_width=True):
            st.session_state.chat_history.append(("customer", message, "", ""))

# Initialize Chatbot
chatbot = AdvancedLoanCollectionChatbot()

# Main Layout
col1, col2 = st.columns([2.5, 1.5])

with col1:
    # Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for i, (role, message, persona, risk) in enumerate(st.session_state.chat_history):
            if role == "customer":
                st.markdown(f"""
                <div class="customer-message">
                    <strong>Customer:</strong> {message}
                    <div class="message-meta">
                        <span class="persona-badge persona-{persona.lower()}">{persona}</span>
                        <span class="risk-{risk.lower()}">{risk} Risk</span>
                        <span>⏰ {datetime.now().strftime('%H:%M')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif role == "agent":
                st.markdown(f"""
                <div class="agent-message">
                    <strong>🤖 AI Agent:</strong> {message}
                    <div class="message-meta">
                        <span>💡 Intelligent Response</span>
                        <span>⏰ {datetime.now().strftime('%H:%M')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="welcome-message">
            <div class="welcome-title">👋 Welcome to AI Collection Assistant</div>
            <p>Start a conversation to see intelligent persona detection and adaptive responses</p>
            <p>💡 Try different emotional tones to experience the AI's adaptability</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input Interface
    st.markdown("### ✍️ Customer Message")
    #user_input = st.text_input(
     #   "",
      #  placeholder="Type your message here (e.g., 'I'm frustrated with my payments' or 'I need help')",
       # key="user_input"
    #)
    user_input = st.text_input(
        "User Input",  # non-empty label
         key="user_input",
         label_visibility="collapsed"  # hides the label from UI but keeps it accessible
    )

    
    # Action Buttons
    col_send, col_clear, col_export = st.columns([2, 1, 1])
    
    with col_send:
        send_button = st.button("🚀 Send Message", type="primary", use_container_width=True)
    
    with col_clear:
        clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
    
    with col_export:
        if st.button("📊 Export Data", use_container_width=True):
            if st.session_state.chat_history:
                chat_data = pd.DataFrame(st.session_state.chat_history, 
                                       columns=['Role', 'Message', 'Persona', 'Risk'])
                st.download_button(
                    "⬇️ Download CSV",
                    chat_data.to_csv(index=False),
                    "chat_history.csv",
                    "text/csv",
                    use_container_width=True
                )

    # Handle button clicks
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.session_start = datetime.now()
        st.rerun()
    
    if send_button and user_input.strip():
        # Prepare customer data
        customer_data = {
            'age': customer_age,
            'income': customer_income,
            'sentiment': customer_sentiment,
            'missed_payments': missed_payments,
            'complaints': complaints,
            'engagement': engagement
        }
        
        # Detect persona and calculate risk
        detected_persona = chatbot.detect_persona_from_message(user_input)
        risk_level, risk_prob = chatbot.calculate_risk_score(customer_data)
        
        # Determine context
        context_keywords = {
            'payment': ['payment', 'pay', 'money', 'amount', 'installment', 'emi'],
            'complaint': ['complaint', 'problem', 'issue', 'wrong', 'error'],
            'closing': ['goodbye', 'bye', 'thanks', 'thank you', 'done'],
            'follow_up': ['follow up', 'next', 'plan', 'schedule']
        }
        
        context = "greeting"
        for ctx, keywords in context_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                context = ctx
                break
        
        # Generate response
        bot_response = chatbot.generate_contextual_response(
            detected_persona, context, risk_level, customer_data
        )
        
        # Add to chat history
        st.session_state.chat_history.append(("customer", user_input, detected_persona, risk_level))
        st.session_state.chat_history.append(("agent", bot_response, detected_persona, risk_level))
        st.session_state.customer_persona = detected_persona
        
        st.rerun()

with col2:
    # Analytics Dashboard
    st.markdown("### 📊 Real-time Analytics")
    
    # Prepare customer data for analysis
    customer_data = {
        'age': customer_age,
        'income': customer_income,
        'sentiment': customer_sentiment,
        'missed_payments': missed_payments,
        'complaints': complaints,
        'engagement': engagement
    }
    
    risk_level, risk_prob = chatbot.calculate_risk_score(customer_data)
    
    # Risk Assessment
    risk_color = "high" if risk_prob >= 0.75 else "medium" if risk_prob >= 0.35 else "low"
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">🎯 Default Risk Score</div>
        <div class="metric-value">{risk_prob:.1%}</div>
        <span class="risk-{risk_color}">🚨 {risk_level} Risk</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Customer Engagement
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">📱 Engagement Level</div>
        <div class="metric-value">{engagement:.1%}</div>
        <div style="background: #e2e8f0; border-radius: 10px; height: 8px; margin-top: 1rem;">
            <div style="background: linear-gradient(90deg, #3b82f6, #1e40af); width: {engagement*100}%; height: 8px; border-radius: 10px; transition: all 0.3s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Financial Health Score
    financial_health = min(1.0, customer_income / 1000000)  # Normalize income
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">💰 Financial Health</div>
        <div class="metric-value">{financial_health:.1%}</div>
        <div style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
            ₹{customer_income:,} annual income
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current Persona
    if st.session_state.chat_history:
        current_persona = st.session_state.customer_persona
        persona_icons = {
            "Cooperative": "😊", "Aggressive": "😠", "Confused": "😕",
            "Evasive": "😐", "Neutral": "😶"
        }
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🎭 Current Persona</div>
            <div class="metric-value">{persona_icons.get(current_persona, '😶')}</div>
            <span class="persona-badge persona-{current_persona.lower()}">{current_persona}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Collection Strategy
        strategy = chatbot.get_collection_strategy(current_persona, risk_level, customer_data)
        st.markdown(f"""
        <div class="strategy-card">
            <div class="strategy-title">🎯 {strategy['approach']}</div>
            <div style="margin-bottom: 1rem;">
                <strong>Timeline:</strong> {strategy['timeline']}
            </div>
            <div>
                <strong>Key Tactics:</strong>
                <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
        """, unsafe_allow_html=True)
        
        for tactic in strategy['tactics'][:3]:  # Show top 3 tactics
            st.markdown(f"<li style='margin: 0.25rem 0;'>{tactic}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div></div>", unsafe_allow_html=True)
    
    # Session Statistics
    if st.session_state.chat_history:
        total_messages = len([msg for msg in st.session_state.chat_history if msg[0] == "customer"])
        session_duration = (datetime.now() - st.session_state.session_start).seconds // 60
        
        # Count persona occurrences
        persona_counts = {}
        for role, msg, persona, risk in st.session_state.chat_history:
            if role == "customer" and persona:
                persona_counts[persona] = persona_counts.get(persona, 0) + 1
        
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">{total_messages}</div>
                <div class="stat-label">Messages</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{session_duration}</div>
                <div class="stat-label">Minutes</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(persona_counts)}</div>
                <div class="stat-label">Personas</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if persona_counts:
            # Persona Distribution Chart
            fig = px.pie(
                values=list(persona_counts.values()),
                names=list(persona_counts.keys()),
                title="Persona Distribution",
                color_discrete_map={
                    'Cooperative': '#3b82f6',
                    'Aggressive': '#ef4444',
                    'Confused': '#f59e0b',
                    'Evasive': '#8b5cf6',
                    'Neutral': '#64748b'
                }
            )
            fig.update_layout(height=300, showlegend=True, font_size=10)
            st.plotly_chart(fig, use_container_width=True)

# Instructions Panel
st.markdown("""
<div class="instructions-panel fade-in">
    <h3 style="color: #1e40af; margin-bottom: 1rem;">🎮 How to Use the AI Collection Assistant</h3>
    
    <div class="instruction-grid">
        <div class="instruction-card">
            <div class="instruction-title">🎛️ Configure Profile</div>
            <p>Adjust customer parameters in the sidebar to simulate different risk profiles and customer types.</p>
        </div>
        
        <div class="instruction-card">
            <div class="instruction-title">💬 Test Conversations</div>
            <p>Type messages or use quick persona tests to see real-time AI adaptation and response strategies.</p>
        </div>
        
        <div class="instruction-card">
            <div class="instruction-title">📊 Monitor Analytics</div>
            <p>Watch the dashboard for live persona detection, risk assessment, and strategy recommendations.</p>
        </div>
        
        <div class="instruction-card">
            <div class="instruction-title">📈 Track Performance</div>
            <p>View conversation statistics, persona distribution, and export data for analysis.</p>
        </div>
    </div>
    
    <div style="margin-top: 2rem; padding: 1.5rem; background: #f8fafc; border-radius: 15px; border-left: 4px solid #3b82f6;">
        <h4 style="color: #1e40af; margin-bottom: 1rem;">💡 Professional Tips for Testing</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
            <div>
                <strong>🎯 Risk Factors:</strong>
                <ul style="margin-top: 0.5rem; color: #64748b;">
                    <li>Higher missed payments = Higher risk</li>
                    <li>Lower income = Increased risk</li>
                    <li>Negative sentiment = Risk amplifier</li>
                    <li>Low engagement = Warning signal</li>
                </ul>
            </div>
            <div>
                <strong>🎭 Persona Testing:</strong>
                <ul style="margin-top: 0.5rem; color: #64748b;">
                    <li>Use emotional language for detection</li>
                    <li>Mix personas in conversations</li>
                    <li>Test edge cases and combinations</li>
                    <li>Observe AI adaptation patterns</li>
                </ul>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 2rem; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 20px; color: white;">
    <h3 style="margin-bottom: 1rem;">🚀 Next-Generation Collection Technology</h3>
    <p style="margin: 0; opacity: 0.9;">Empowering financial institutions with AI-driven, empathetic collection strategies</p>
    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        Built with Streamlit • Powered by Advanced ML • Designed for Professional Excellence
    </div>
</div>
""", unsafe_allow_html=True)