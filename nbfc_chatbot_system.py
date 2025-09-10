# NBFC Loan Collection Analytics Chatbot System
# Complete end-to-end solution for loan collection management

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

class NBFCLoanCollectionSystem:
    """
    Complete NBFC Loan Collection System with ML prediction, 
    persona detection, strategy recommendation, and chatbot functionality
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.conversation_history = []
        self.customer_profile = {}
        self.current_persona = "neutral"
        self.risk_score = 0.0
        
        # Initialize persona keywords and sentiment patterns
        self._init_persona_patterns()
        
        # Initialize collection strategies
        self._init_collection_strategies()
        
    def _init_persona_patterns(self):
        """Initialize patterns for persona detection"""
        self.persona_patterns = {
            'cooperative': {
                'keywords': ['understand', 'sorry', 'help', 'willing', 'try', 'pay', 'arrange', 'plan', 'agree'],
                'sentiment_range': (0.2, 1.0),
                'response_indicators': ['yes', 'okay', 'sure', 'will do', 'thank you']
            },
            'evasive': {
                'keywords': ['busy', 'later', 'call back', 'not now', 'traveling', 'meeting', 'cant talk'],
                'sentiment_range': (-0.3, 0.3),
                'response_indicators': ['maybe', 'perhaps', 'not sure', 'difficult', 'complicated']
            },
            'aggressive': {
                'keywords': ['harassment', 'legal', 'lawyer', 'complain', 'stop calling', 'annoying', 'rude'],
                'sentiment_range': (-1.0, -0.5),
                'response_indicators': ['no', 'never', 'dont', 'stop', 'enough']
            },
            'confused': {
                'keywords': ['dont understand', 'what', 'how', 'why', 'explain', 'unclear', 'confused'],
                'sentiment_range': (-0.2, 0.4),
                'response_indicators': ['huh', 'what do you mean', 'i dont get it', 'clarify']
            },
            'financial_distress': {
                'keywords': ['lost job', 'medical', 'emergency', 'crisis', 'difficult time', 'struggling'],
                'sentiment_range': (-0.6, 0.1),
                'response_indicators': ['cant afford', 'no money', 'financial problems']
            }
        }
    
    def _init_collection_strategies(self):
        """Initialize collection strategies for different personas and risk levels"""
        self.collection_strategies = {
            'cooperative': {
                'low_risk': {
                    'approach': 'Empathetic & Supportive',
                    'tone': 'friendly',
                    'tactics': ['Payment reminder', 'Flexible payment options', 'Appreciation for cooperation'],
                    'timeline': '7-14 days',
                    'escalation': 'Gentle follow-up'
                },
                'medium_risk': {
                    'approach': 'Structured Payment Plan',
                    'tone': 'professional',
                    'tactics': ['Formal payment plan', 'Regular check-ins', 'Incentives for timely payment'],
                    'timeline': '5-10 days',
                    'escalation': 'Account review'
                },
                'high_risk': {
                    'approach': 'Immediate Action Required',
                    'tone': 'firm but fair',
                    'tactics': ['Urgent payment request', 'Settlement options', 'Clear consequences'],
                    'timeline': '2-5 days',
                    'escalation': 'Legal notice'
                }
            },
            'evasive': {
                'low_risk': {
                    'approach': 'Persistent but Respectful',
                    'tone': 'professional',
                    'tactics': ['Multiple contact attempts', 'Clear deadlines', 'Written communication'],
                    'timeline': '10-15 days',
                    'escalation': 'Formal notice'
                },
                'medium_risk': {
                    'approach': 'Firm Communication',
                    'tone': 'assertive',
                    'tactics': ['Scheduled calls', 'Account freeze warning', 'Payment deadline'],
                    'timeline': '5-8 days',
                    'escalation': 'Account restriction'
                },
                'high_risk': {
                    'approach': 'Immediate Escalation',
                    'tone': 'firm',
                    'tactics': ['Final notice', 'Legal action warning', 'Asset verification'],
                    'timeline': '1-3 days',
                    'escalation': 'Legal proceedings'
                }
            },
            'aggressive': {
                'low_risk': {
                    'approach': 'De-escalation & Professional',
                    'tone': 'calm',
                    'tactics': ['Acknowledge concerns', 'Explain process', 'Offer alternatives'],
                    'timeline': '14-21 days',
                    'escalation': 'Supervisor involvement'
                },
                'medium_risk': {
                    'approach': 'Formal & Documented',
                    'tone': 'professional',
                    'tactics': ['Written communication only', 'Clear documentation', 'Compliance focus'],
                    'timeline': '7-10 days',
                    'escalation': 'Legal review'
                },
                'high_risk': {
                    'approach': 'Legal & Compliance Focused',
                    'tone': 'formal',
                    'tactics': ['Legal notice', 'Compliance documentation', 'Third-party mediation'],
                    'timeline': '3-5 days',
                    'escalation': 'Court proceedings'
                }
            },
            'confused': {
                'low_risk': {
                    'approach': 'Educational & Patient',
                    'tone': 'helpful',
                    'tactics': ['Simple explanations', 'Step-by-step guidance', 'Educational materials'],
                    'timeline': '10-14 days',
                    'escalation': 'Additional support'
                },
                'medium_risk': {
                    'approach': 'Guided Support',
                    'tone': 'supportive',
                    'tactics': ['Personal assistance', 'Simplified options', 'Regular guidance'],
                    'timeline': '7-10 days',
                    'escalation': 'Specialized support'
                },
                'high_risk': {
                    'approach': 'Intensive Support',
                    'tone': 'patient but urgent',
                    'tactics': ['One-on-one guidance', 'Immediate assistance', 'Simplified process'],
                    'timeline': '3-7 days',
                    'escalation': 'Direct intervention'
                }
            },
            'financial_distress': {
                'low_risk': {
                    'approach': 'Empathetic & Flexible',
                    'tone': 'compassionate',
                    'tactics': ['Hardship assessment', 'Flexible terms', 'Support resources'],
                    'timeline': '21-30 days',
                    'escalation': 'Hardship program'
                },
                'medium_risk': {
                    'approach': 'Structured Hardship Support',
                    'tone': 'understanding',
                    'tactics': ['Income verification', 'Reduced payments', 'Extended timeline'],
                    'timeline': '14-21 days',
                    'escalation': 'Hardship committee'
                },
                'high_risk': {
                    'approach': 'Emergency Assistance',
                    'tone': 'supportive',
                    'tactics': ['Emergency payment plan', 'Partial settlement', 'Financial counseling'],
                    'timeline': '7-14 days',
                    'escalation': 'Workout arrangement'
                }
            }
        }
    
    def preprocess_data(self, df):
        """Preprocess the dataset for machine learning"""
        data = df.copy()
        
        # Initialize label encoders
        self.label_encoders['location'] = LabelEncoder()
        self.label_encoders['employment'] = LabelEncoder()
        self.label_encoders['loantype'] = LabelEncoder()
        
        # Encode categorical variables
        data['Location_encoded'] = self.label_encoders['location'].fit_transform(data['Location'])
        data['EmploymentStatus_encoded'] = self.label_encoders['employment'].fit_transform(data['EmploymentStatus'])
        data['LoanType_encoded'] = self.label_encoders['loantype'].fit_transform(data['LoanType'])
        
        # Create additional features
        data['LTV_ratio'] = data['LoanAmount'] / data['Income']
        data['Payment_behavior_score'] = (data['MissedPayments'] * 2 + data['DelaysDays'] / 30 + data['PartialPayments'])
        data['Engagement_score'] = data['AppUsageFrequency'] + (data['WebsiteVisits'] / 50)
        data['Risk_indicator'] = (data['MissedPayments'] > 2).astype(int)
        
        # Feature selection
        self.feature_columns = ['Age', 'Income', 'LoanAmount', 'TenureMonths', 'InterestRate',
                               'MissedPayments', 'DelaysDays', 'PartialPayments', 'InteractionAttempts',
                               'SentimentScore', 'ResponseTimeHours', 'AppUsageFrequency', 
                               'WebsiteVisits', 'Complaints', 'Location_encoded', 
                               'EmploymentStatus_encoded', 'LoanType_encoded', 'LTV_ratio',
                               'Payment_behavior_score', 'Engagement_score', 'Risk_indicator']
        
        X = data[self.feature_columns]
        y = data['Target']
        
        return X, y
    
    def train_model(self, df):
        """Train the machine learning model"""
        X, y = self.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest model (best performing)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate and print performance
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        return accuracy, roc_auc
    
    def predict_risk(self, customer_data: Dict) -> float:
        """Predict risk probability for a customer"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare input data
        input_data = self._prepare_input_data(customer_data)
        
        # Make prediction
        risk_probability = self.model.predict_proba([input_data])[0][1]
        
        return risk_probability
    
    def _prepare_input_data(self, customer_data: Dict) -> List:
        """Prepare customer data for model prediction"""
        # Handle categorical encoding
        location_encoded = self._safe_encode('location', customer_data.get('Location', 'Urban'))
        employment_encoded = self._safe_encode('employment', customer_data.get('EmploymentStatus', 'Salaried'))
        loantype_encoded = self._safe_encode('loantype', customer_data.get('LoanType', 'Personal'))
        
        # Calculate derived features
        ltv_ratio = customer_data.get('LoanAmount', 500000) / customer_data.get('Income', 1000000)
        payment_behavior_score = (customer_data.get('MissedPayments', 0) * 2 + 
                                customer_data.get('DelaysDays', 0) / 30 + 
                                customer_data.get('PartialPayments', 0))
        engagement_score = (customer_data.get('AppUsageFrequency', 0.5) + 
                          customer_data.get('WebsiteVisits', 25) / 50)
        risk_indicator = 1 if customer_data.get('MissedPayments', 0) > 2 else 0
        
        # Prepare input array in the same order as feature_columns
        input_data = [
            customer_data.get('Age', 40),
            customer_data.get('Income', 1000000),
            customer_data.get('LoanAmount', 500000),
            customer_data.get('TenureMonths', 36),
            customer_data.get('InterestRate', 10.5),
            customer_data.get('MissedPayments', 0),
            customer_data.get('DelaysDays', 0),
            customer_data.get('PartialPayments', 0),
            customer_data.get('InteractionAttempts', 0),
            customer_data.get('SentimentScore', 0.0),
            customer_data.get('ResponseTimeHours', 24.0),
            customer_data.get('AppUsageFrequency', 0.5),
            customer_data.get('WebsiteVisits', 25),
            customer_data.get('Complaints', 0),
            location_encoded,
            employment_encoded,
            loantype_encoded,
            ltv_ratio,
            payment_behavior_score,
            engagement_score,
            risk_indicator
        ]
        
        return input_data
    
    def _safe_encode(self, encoder_type: str, value: str) -> int:
        """Safely encode categorical values"""
        try:
            return self.label_encoders[encoder_type].transform([value])[0]
        except (ValueError, KeyError):
            # Return default encoding for unknown values
            return 0
    
    def detect_persona(self, message: str, sentiment_score: float = None) -> str:
        """Detect customer persona from message and sentiment"""
        message_lower = message.lower()
        
        # Count keyword matches for each persona
        persona_scores = {}
        
        for persona, patterns in self.persona_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in patterns['keywords']:
                if keyword in message_lower:
                    score += 1
            
            # Response indicator matching
            for indicator in patterns['response_indicators']:
                if indicator in message_lower:
                    score += 2
            
            # Sentiment range matching
            if sentiment_score is not None:
                sent_min, sent_max = patterns['sentiment_range']
                if sent_min <= sentiment_score <= sent_max:
                    score += 1
            
            persona_scores[persona] = score
        
        # Return persona with highest score, default to neutral
        if max(persona_scores.values()) > 0:
            return max(persona_scores.keys(), key=persona_scores.get)
        else:
            return 'neutral'
    
    def get_collection_strategy(self, persona: str, risk_level: str) -> Dict:
        """Get collection strategy based on persona and risk level"""
        # Default to cooperative if persona not found
        persona_strategies = self.collection_strategies.get(persona, self.collection_strategies['cooperative'])
        
        # Get strategy for risk level
        strategy = persona_strategies.get(risk_level, persona_strategies['medium_risk'])
        
        return strategy
    
    def get_risk_level(self, risk_probability: float) -> str:
        """Convert risk probability to risk level"""
        if risk_probability < 0.3:
            return 'low_risk'
        elif risk_probability < 0.7:
            return 'medium_risk'
        else:
            return 'high_risk'
    
    def generate_response(self, user_message: str, customer_data: Dict = None) -> Dict:
        """Generate chatbot response based on user message and customer data"""
        
        # Update customer profile if provided
        if customer_data:
            self.customer_profile.update(customer_data)
        
        # Detect persona from message
        sentiment_score = customer_data.get('SentimentScore', 0.0) if customer_data else 0.0
        detected_persona = self.detect_persona(user_message, sentiment_score)
        self.current_persona = detected_persona
        
        # Predict risk if we have customer data
        if self.customer_profile and self.model:
            try:
                self.risk_score = self.predict_risk(self.customer_profile)
                risk_level = self.get_risk_level(self.risk_score)
            except:
                self.risk_score = 0.5
                risk_level = 'medium_risk'
        else:
            self.risk_score = 0.5
            risk_level = 'medium_risk'
        
        # Get collection strategy
        strategy = self.get_collection_strategy(detected_persona, risk_level)
        
        # Generate contextual response
        response_text = self._generate_contextual_response(user_message, detected_persona, strategy)
        
        # Store conversation
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'detected_persona': detected_persona,
            'risk_score': self.risk_score,
            'risk_level': risk_level,
            'bot_response': response_text,
            'strategy': strategy
        }
        
        self.conversation_history.append(conversation_entry)
        
        return {
            'response': response_text,
            'persona': detected_persona,
            'risk_score': self.risk_score,
            'risk_level': risk_level,
            'strategy': strategy,
            'conversation_id': len(self.conversation_history)
        }
    
    def _generate_contextual_response(self, user_message: str, persona: str, strategy: Dict) -> str:
        """Generate contextual response based on persona and strategy"""
        
        tone = strategy.get('tone', 'professional')
        approach = strategy.get('approach', 'Standard')
        
        # Base responses by persona
        responses = {
            'cooperative': {
                'greeting': "Thank you for taking the time to speak with us today. I appreciate your cooperation.",
                'payment_discussion': f"I understand your willingness to resolve this matter. Based on your account, we can work together on a {approach.lower()} approach.",
                'solution': "Let's explore some flexible payment options that work for your current situation."
            },
            'evasive': {
                'greeting': "I understand you may be busy, but it's important we discuss your account today.",
                'payment_discussion': f"We need to address your payment situation directly. Our {approach.lower()} requires your immediate attention.",
                'solution': "I need a specific commitment from you today. When can you make your next payment?"
            },
            'aggressive': {
                'greeting': "I understand your concerns. Let me explain the situation clearly and professionally.",
                'payment_discussion': f"We are committed to following proper procedures. Our {approach.lower()} ensures compliance with all regulations.",
                'solution': "Let's focus on finding a mutually acceptable resolution that addresses your concerns."
            },
            'confused': {
                'greeting': "I'm here to help clarify any questions you may have about your account.",
                'payment_discussion': f"Let me explain this step by step. We're implementing a {approach.lower()} to make this easier to understand.",
                'solution': "Would you like me to break down your payment options in simple terms?"
            },
            'financial_distress': {
                'greeting': "I understand you may be going through a difficult time. We're here to help find a solution.",
                'payment_discussion': f"We recognize your current situation and want to work with you using our {approach.lower()}.",
                'solution': "Let's discuss some hardship options that might be available to help you through this period."
            }
        }
        
        # Get appropriate response set or default to professional
        persona_responses = responses.get(persona, responses['cooperative'])
        
        # Simple keyword-based response selection
        user_lower = user_message.lower()
        
        if any(word in user_lower for word in ['hello', 'hi', 'good morning', 'good afternoon']):
            return persona_responses['greeting']
        elif any(word in user_lower for word in ['payment', 'pay', 'money', 'amount', 'due']):
            return persona_responses['payment_discussion']
        elif any(word in user_lower for word in ['help', 'option', 'what can', 'solution']):
            return persona_responses['solution']
        else:
            # Default response with strategy info
            return f"Thank you for your message. {persona_responses['greeting']} {persona_responses['solution']}"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def export_conversation_history(self, filename: str = None) -> str:
        """Export conversation history to CSV"""
        if not filename:
            filename = f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        if self.conversation_history:
            df = pd.DataFrame(self.conversation_history)
            df.to_csv(filename, index=False)
            return filename
        else:
            return "No conversation history to export"
    
    def get_analytics_summary(self) -> Dict:
        """Get analytics summary of conversations"""
        if not self.conversation_history:
            return {"message": "No conversation data available"}
        
        df = pd.DataFrame(self.conversation_history)
        
        persona_distribution = df['detected_persona'].value_counts().to_dict()
        risk_level_distribution = df['risk_level'].value_counts().to_dict()
        avg_risk_score = df['risk_score'].mean()
        
        return {
            'total_conversations': len(self.conversation_history),
            'persona_distribution': persona_distribution,
            'risk_level_distribution': risk_level_distribution,
            'average_risk_score': avg_risk_score,
            'most_common_persona': max(persona_distribution.keys(), key=persona_distribution.get) if persona_distribution else None
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    nbfc_system = NBFCLoanCollectionSystem()
    
    # Load and train with sample data (you would load your actual dataset)
    print("NBFC Loan Collection Analytics System")
    print("=" * 50)
    
    # Example customer data
    sample_customer = {
        'Age': 35,
        'Income': 800000,
        'Location': 'Urban',
        'EmploymentStatus': 'Salaried',
        'LoanAmount': 500000,
        'TenureMonths': 36,
        'InterestRate': 12.5,
        'LoanType': 'Personal',
        'MissedPayments': 2,
        'DelaysDays': 45,
        'PartialPayments': 1,
        'InteractionAttempts': 3,
        'SentimentScore': -0.3,
        'ResponseTimeHours': 48.0,
        'AppUsageFrequency': 0.6,
        'WebsiteVisits': 15,
        'Complaints': 1
    }
    
    print("Sample Customer Profile:", sample_customer)
    print()
    
    # Test conversation examples
    test_messages = [
        "Hello, I received a call about my loan payment",
        "I'm sorry, I've been having financial difficulties lately",
        "Stop calling me! This is harassment!",
        "I don't understand what you're asking for",
        "I'm willing to work out a payment plan"
    ]
    
    print("Testing Conversation Examples:")
    print("-" * 30)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\\nTest {i}:")
        print(f"User: {message}")
        
        # Generate response (without trained model for demo)
        response_data = nbfc_system.generate_response(message, sample_customer)
        
        print(f"Bot: {response_data['response']}")
        print(f"Detected Persona: {response_data['persona']}")
        print(f"Risk Level: {response_data['risk_level']}")
        print(f"Strategy: {response_data['strategy']['approach']}")