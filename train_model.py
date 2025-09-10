# Training Script and Model Initialization
# This script trains the ML model and saves it for use in the chatbot

import pandas as pd
import numpy as np
from nbfc_chatbot_system import NBFCLoanCollectionSystem
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_and_save_model():
    """Train the model with the provided dataset and save it"""
    
    print("🏦 NBFC Loan Collection ML Model Training")
    print("=" * 50)
    
    # Initialize the system
    nbfc_system = NBFCLoanCollectionSystem()
    
    # Load the dataset
    try:
        df = pd.read_csv('Analytics_loan_collection_dataset.csv')
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("❌ Dataset file not found. Please ensure 'Analytics_loan_collection_dataset.csv' is in the current directory.")
        return None
    
    # Train the model
    print("\\n🔄 Training machine learning model...")
    accuracy, roc_auc = nbfc_system.train_model(df)
    
    print(f"\\n✅ Model Training Complete!")
    print(f"📊 Model Performance:")
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • ROC-AUC: {roc_auc:.4f}")
    
    # Save the trained model
    model_filename = 'nbfc_trained_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': nbfc_system.model,
            'label_encoders': nbfc_system.label_encoders,
            'feature_columns': nbfc_system.feature_columns
        }, f)
    
    print(f"\\n💾 Model saved as '{model_filename}'")
    
    return nbfc_system

def demonstrate_conversation_flows():
    """Demonstrate different conversation flows for each persona"""
    
    print("\\n\\n💬 CONVERSATION FLOW DEMONSTRATIONS")
    print("=" * 50)
    
    # Initialize system
    nbfc_system = NBFCLoanCollectionSystem()
    
    # Sample customer profiles for different scenarios
    customers = {
        'cooperative_customer': {
            'Age': 35, 'Income': 800000, 'Location': 'Urban', 'EmploymentStatus': 'Salaried',
            'LoanAmount': 500000, 'TenureMonths': 36, 'InterestRate': 12.5, 'LoanType': 'Personal',
            'MissedPayments': 1, 'DelaysDays': 15, 'PartialPayments': 0, 'InteractionAttempts': 1,
            'SentimentScore': 0.3, 'ResponseTimeHours': 12.0, 'AppUsageFrequency': 0.8,
            'WebsiteVisits': 25, 'Complaints': 0
        },
        'evasive_customer': {
            'Age': 42, 'Income': 600000, 'Location': 'Suburban', 'EmploymentStatus': 'Self-Employed',
            'LoanAmount': 700000, 'TenureMonths': 48, 'InterestRate': 14.0, 'LoanType': 'Business',
            'MissedPayments': 3, 'DelaysDays': 60, 'PartialPayments': 2, 'InteractionAttempts': 5,
            'SentimentScore': -0.1, 'ResponseTimeHours': 48.0, 'AppUsageFrequency': 0.3,
            'WebsiteVisits': 8, 'Complaints': 1
        },
        'aggressive_customer': {
            'Age': 38, 'Income': 1200000, 'Location': 'Urban', 'EmploymentStatus': 'Salaried',
            'LoanAmount': 800000, 'TenureMonths': 24, 'InterestRate': 11.5, 'LoanType': 'Auto',
            'MissedPayments': 2, 'DelaysDays': 30, 'PartialPayments': 1, 'InteractionAttempts': 3,
            'SentimentScore': -0.7, 'ResponseTimeHours': 6.0, 'AppUsageFrequency': 0.5,
            'WebsiteVisits': 12, 'Complaints': 2
        },
        'confused_customer': {
            'Age': 28, 'Income': 400000, 'Location': 'Rural', 'EmploymentStatus': 'Student',
            'LoanAmount': 300000, 'TenureMonths': 60, 'InterestRate': 13.5, 'LoanType': 'Education',
            'MissedPayments': 2, 'DelaysDays': 45, 'PartialPayments': 1, 'InteractionAttempts': 2,
            'SentimentScore': 0.1, 'ResponseTimeHours': 24.0, 'AppUsageFrequency': 0.4,
            'WebsiteVisits': 5, 'Complaints': 0
        },
        'financial_distress_customer': {
            'Age': 45, 'Income': 500000, 'Location': 'Suburban', 'EmploymentStatus': 'Unemployed',
            'LoanAmount': 600000, 'TenureMonths': 36, 'InterestRate': 15.0, 'LoanType': 'Personal',
            'MissedPayments': 4, 'DelaysDays': 90, 'PartialPayments': 3, 'InteractionAttempts': 6,
            'SentimentScore': -0.4, 'ResponseTimeHours': 72.0, 'AppUsageFrequency': 0.2,
            'WebsiteVisits': 3, 'Complaints': 1
        }
    }
    
    # Conversation scenarios
    conversation_flows = {
        'cooperative_customer': [
            "Hello, I received a call about my loan payment. I'm sorry for the delay.",
            "Yes, I understand. I'm willing to make the payment. What are my options?",
            "That sounds reasonable. I can arrange for the payment by Friday.",
            "Thank you for being understanding. I'll set up the payment today."
        ],
        'evasive_customer': [
            "Hi, I'm quite busy right now. Can we talk later?",
            "I'm traveling for work this week. Maybe we can discuss this next week?",
            "It's complicated right now. I'll call you back when I have time.",
            "I'm in a meeting. Can you send me the details via email?"
        ],
        'aggressive_customer': [
            "Stop calling me! This is harassment!",
            "I'm going to complain to your manager about this constant calling.",
            "I know my rights. You can't keep bothering me like this.",
            "I'm thinking of getting a lawyer involved if this continues."
        ],
        'confused_customer': [
            "I don't understand what you're asking for. Can you explain?",
            "What exactly do I need to pay? I'm confused about the amount.",
            "How did you calculate this? I don't get the breakdown.",
            "Can you help me understand what my options are?"
        ],
        'financial_distress_customer': [
            "I lost my job last month and I'm struggling financially.",
            "I want to pay but I'm going through a difficult time right now.",
            "I had a medical emergency that wiped out my savings.",
            "Is there any way you can help me with a payment plan?"
        ]
    }
    
    # Demonstrate each conversation flow
    for customer_type, customer_data in customers.items():
        print(f"\\n\\n🎭 {customer_type.replace('_', ' ').title()} Conversation Flow")
        print("-" * 60)
        print(f"Customer Profile: {customer_data['Age']} years old, {customer_data['EmploymentStatus']}, {customer_data['Location']}")
        print(f"Loan: ₹{customer_data['LoanAmount']:,} ({customer_data['LoanType']})")
        print(f"Payment Issues: {customer_data['MissedPayments']} missed payments, {customer_data['DelaysDays']} delay days")
        print()
        
        messages = conversation_flows[customer_type]
        
        for i, message in enumerate(messages, 1):
            print(f"Turn {i}:")
            print(f"👤 Customer: {message}")
            
            # Generate AI response
            response_data = nbfc_system.generate_response(message, customer_data)
            
            print(f"🤖 AI Assistant: {response_data['response']}")
            print(f"   📊 Detected Persona: {response_data['persona']}")
            print(f"   ⚠️  Risk Level: {response_data['risk_level']}")
            print(f"   🎯 Strategy: {response_data['strategy']['approach']}")
            print()

def create_sample_training_data():
    """Create sample training data for demonstration purposes"""
    
    print("\\n\\n📊 CREATING SAMPLE TRAINING DATA")
    print("=" * 50)
    
    # Sample conversation data with labels
    sample_conversations = [
        {
            'message': 'I apologize for the delay. I can pay tomorrow.',
            'persona': 'cooperative',
            'sentiment': 0.4,
            'keywords': ['apologize', 'delay', 'pay', 'tomorrow']
        },
        {
            'message': 'I am busy right now. Can you call me later?',
            'persona': 'evasive',
            'sentiment': 0.0,
            'keywords': ['busy', 'later', 'call']
        },
        {
            'message': 'Stop harassing me! I will file a complaint.',
            'persona': 'aggressive',
            'sentiment': -0.8,
            'keywords': ['stop', 'harassing', 'complaint']
        },
        {
            'message': 'I do not understand what you are asking for.',
            'persona': 'confused',
            'sentiment': 0.1,
            'keywords': ['not understand', 'asking']
        },
        {
            'message': 'I lost my job and cannot afford the payment.',
            'persona': 'financial_distress',
            'sentiment': -0.3,
            'keywords': ['lost job', 'cannot afford', 'payment']
        }
    ]
    
    sample_df = pd.DataFrame(sample_conversations)
    sample_df.to_csv('sample_conversation_training_data.csv', index=False)
    print("✅ Sample conversation training data created: 'sample_conversation_training_data.csv'")
    
    return sample_df

def create_deployment_guide():
    """Create a deployment guide for the system"""
    
    guide_content = '''
# 🏦 NBFC AI Collection Assistant - Deployment Guide

## 📋 System Requirements

### Python Dependencies
```
pip install pandas numpy scikit-learn streamlit plotly
```

### Files Required
- `nbfc_chatbot_system.py` - Core system implementation
- `streamlit_app.py` - Web interface
- `Analytics_loan_collection_dataset.csv` - Training dataset
- `train_model.py` - Model training script

## 🚀 Quick Start

### 1. Train the Model
```bash
python train_model.py
```

### 2. Launch the Web Application
```bash
streamlit run streamlit_app.py
```

### 3. Access the Application
Open your browser and go to: `http://localhost:8501`

## 🔧 Configuration

### Model Configuration
- Adjust `n_estimators` in RandomForestClassifier for different accuracy/speed trade-offs
- Modify `feature_columns` to include/exclude specific features
- Update `persona_patterns` to customize persona detection

### UI Customization
- Modify CSS in `streamlit_app.py` for different styling
- Add new tabs or sections as needed
- Customize color schemes and layouts

## 📊 Features

### 1. Predictive Modeling
- Random Forest classifier for default prediction
- Feature importance analysis
- Risk scoring (0-100%)

### 2. Persona Detection
- Rule-based classification using keywords and sentiment
- 5 main personas: Cooperative, Evasive, Aggressive, Confused, Financial Distress
- Real-time adaptation

### 3. Strategy Recommendation
- Dynamic strategy selection based on persona and risk level
- Customizable collection approaches
- Timeline and escalation guidance

### 4. Chatbot Interface
- Natural conversation flow
- Context-aware responses
- Professional tone adaptation

### 5. Analytics Dashboard
- Real-time risk assessment
- Conversation analytics
- Strategy effectiveness tracking
- Export functionality

## 🛠️ Customization

### Adding New Personas
1. Update `persona_patterns` in `NBFCLoanCollectionSystem`
2. Add corresponding strategies in `collection_strategies`
3. Update response templates in `_generate_contextual_response`

### Modifying Risk Calculation
1. Adjust feature weights in the ML model
2. Update risk thresholds in `get_risk_level`
3. Customize feature engineering in `_prepare_input_data`

### Enhancing Strategies
1. Add new strategy types in `collection_strategies`
2. Implement strategy effectiveness tracking
3. Add automated escalation rules

## 📈 Performance Optimization

### Model Performance
- Use cross-validation for better model selection
- Implement ensemble methods for improved accuracy
- Add feature selection techniques

### System Performance
- Cache trained models for faster response times
- Implement database storage for conversation history
- Add API endpoints for integration

## 🔒 Security Considerations

### Data Protection
- Implement encryption for customer data
- Add user authentication
- Ensure GDPR compliance

### System Security
- Input validation and sanitization
- Rate limiting for API calls
- Secure model storage

## 📞 Support and Maintenance

### Regular Updates
- Retrain models with new data monthly
- Update persona patterns based on new conversation types
- Monitor strategy effectiveness and adjust

### Monitoring
- Track model performance metrics
- Monitor conversation success rates
- Log system errors and performance issues

## 🤝 Integration

### CRM Integration
- API endpoints for customer data retrieval
- Real-time updates to customer records
- Automated workflow integration

### Communication Channels
- SMS integration for automated messages
- Email templates for formal communications
- Call center integration

For technical support, refer to the documentation or contact the development team.
'''
    
    with open('DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    print("\\n📋 Deployment guide created: 'DEPLOYMENT_GUIDE.md'")

if __name__ == "__main__":
    # Run all setup functions
    print("🏦 NBFC AI Collection Assistant - Complete Setup")
    print("=" * 60)
    
    # Train and save the model
    trained_system = train_and_save_model()
    
    # Create sample training data
    create_sample_training_data()
    
    # Demonstrate conversation flows
    demonstrate_conversation_flows()
    
    # Create deployment guide
    create_deployment_guide()
    
    print("\\n\\n✅ SETUP COMPLETE!")
    print("=" * 60)
    print("🚀 Next Steps:")
    print("1. Review the generated deployment guide")
    print("2. Launch the Streamlit app: streamlit run streamlit_app.py")
    print("3. Test different conversation scenarios")
    print("4. Customize personas and strategies as needed")
    print("\\n🎯 Your NBFC AI Collection Assistant is ready to use!")
