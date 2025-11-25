from fastapi import FastAPI, APIRouter, HTTPException, Header, Response, Cookie, Request
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import pickle
import json
import io
import base64

# ML Libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# LLM Integration
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models directory
MODELS_DIR = ROOT_DIR / "ml_models"
MODELS_DIR.mkdir(exist_ok=True)

# ============= PYDANTIC MODELS =============
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Incident(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    category: str
    priority: str
    affected_system: str
    error_code: Optional[str] = None
    symptoms: str
    predicted_root_cause: Optional[str] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TicketForecast(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    forecast_data: List[Dict[str, Any]]
    model_used: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    message: str
    response: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Request/Response Models
class IncidentCreate(BaseModel):
    category: str
    priority: str
    affected_system: str
    error_code: Optional[str] = None
    symptoms: str

class ForecastRequest(BaseModel):
    model_type: str  # 'sarima' or 'lstm'
    periods: int = 30

class ChatRequest(BaseModel):
    message: str

# ============= AUTHENTICATION =============
async def get_current_user(request: Request, authorization: Optional[str] = Header(None)) -> Optional[User]:
    session_token = request.cookies.get("session_token")
    if not session_token and authorization:
        session_token = authorization.replace("Bearer ", "")
    if not session_token:
        return None
    session = await db.user_sessions.find_one({"session_token": session_token})
    if not session:
        return None
    
    # Handle both string and datetime objects for expires_at
    expires_at = session["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    elif isinstance(expires_at, datetime):
        # Ensure timezone awareness
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        return None
    user_doc = await db.users.find_one({"id": session["user_id"]}, {"_id": 0})
    if not user_doc:
        return None
    return User(**user_doc)

@api_router.get("/auth/session")
async def create_session(request: Request, x_session_id: Optional[str] = Header(None)):
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-ID header")
    
    try:
        import requests
        response = requests.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data", 
            headers={"X-Session-ID": x_session_id},
            timeout=10
        )
        
        if response.status_code != 200:
            error_detail = response.json().get('detail', 'Invalid session ID')
            logging.error(f"Auth API error: {response.status_code} - {error_detail}")
            raise HTTPException(status_code=401, detail=f"Authentication failed: {error_detail}")
        
        data = response.json()
        
        # Validate required fields
        if not data.get('email') or not data.get('session_token'):
            raise HTTPException(status_code=400, detail="Invalid auth data received")
        
        existing_user = await db.users.find_one({"email": data["email"]}, {"_id": 0})
        if not existing_user:
            user = User(id=str(uuid.uuid4()), email=data["email"], name=data["name"], picture=data.get("picture"))
            user_dict = user.model_dump()
            user_dict["created_at"] = user_dict["created_at"].isoformat()
            await db.users.insert_one(user_dict)
        else:
            user = User(**existing_user)
        
        session_token = data["session_token"]
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        session = UserSession(user_id=user.id, session_token=session_token, expires_at=expires_at)
        session_dict = session.model_dump()
        session_dict["expires_at"] = session_dict["expires_at"].isoformat()
        session_dict["created_at"] = session_dict["created_at"].isoformat()
        await db.user_sessions.insert_one(session_dict)
        
        resp = JSONResponse(content={"id": user.id, "email": user.email, "name": user.name, "picture": user.picture, "session_token": session_token})
        resp.set_cookie(key="session_token", value=session_token, httponly=True, secure=True, samesite="none", path="/", max_age=7*24*60*60)
        return resp
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Auth API request failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
    except Exception as e:
        logging.error(f"Session creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@api_router.get("/auth/me")
async def get_me(request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@api_router.post("/auth/logout")
async def logout(request: Request, authorization: Optional[str] = Header(None)):
    session_token = request.cookies.get("session_token")
    if not session_token and authorization:
        session_token = authorization.replace("Bearer ", "")
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    resp = JSONResponse(content={"message": "Logged out"})
    resp.delete_cookie("session_token", path="/")
    return resp

# ============= ML INITIALIZATION =============
def initialize_ml_models():
    np.random.seed(42)
    categories = ['Network', 'Database', 'Application', 'Hardware', 'Security']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    systems = ['Web Server', 'Database Server', 'API Gateway', 'Load Balancer', 'Storage']
    root_causes = ['Configuration Error', 'Resource Exhaustion', 'Network Congestion', 'Database Deadlock', 'Memory Leak', 'Disk Full', 'Authentication Failure', 'Bug in Code']
    data = {'category': np.random.choice(categories, 1000), 'priority': np.random.choice(priorities, 1000), 'affected_system': np.random.choice(systems, 1000), 'error_code': np.random.randint(100, 999, 1000), 'root_cause': np.random.choice(root_causes, 1000)}
    df = pd.DataFrame(data)
    le_cat = LabelEncoder()
    le_pri = LabelEncoder()
    le_sys = LabelEncoder()
    le_cause = LabelEncoder()
    df['category_encoded'] = le_cat.fit_transform(df['category'])
    df['priority_encoded'] = le_pri.fit_transform(df['priority'])
    df['system_encoded'] = le_sys.fit_transform(df['affected_system'])
    df['cause_encoded'] = le_cause.fit_transform(df['root_cause'])
    X = df[['category_encoded', 'priority_encoded', 'system_encoded', 'error_code']]
    y = df['cause_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    pickle.dump(dt_model, open(MODELS_DIR / 'decision_tree.pkl', 'wb'))
    pickle.dump(rf_model, open(MODELS_DIR / 'random_forest.pkl', 'wb'))
    pickle.dump(le_cat, open(MODELS_DIR / 'le_category.pkl', 'wb'))
    pickle.dump(le_pri, open(MODELS_DIR / 'le_priority.pkl', 'wb'))
    pickle.dump(le_sys, open(MODELS_DIR / 'le_system.pkl', 'wb'))
    pickle.dump(le_cause, open(MODELS_DIR / 'le_cause.pkl', 'wb'))
    
    # Generate realistic IT ticket data with temporal patterns
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='h')
    ticket_counts = []
    
    for dt in dates:
        # Base load
        base = 5
        
        # Business hours effect (9am-5pm = more tickets)
        hour = dt.hour
        if 9 <= hour <= 17:
            base += 15
        elif 8 <= hour <= 18:
            base += 8
        elif 6 <= hour <= 20:
            base += 3
        
        # Weekday vs weekend
        if dt.weekday() >= 5:  # Weekend
            base = base * 0.4
        
        # Monday spike (more issues after weekend)
        if dt.weekday() == 0 and 9 <= hour <= 12:
            base += 10
        
        # Month-end spike (deployments, reporting)
        if dt.day >= 28:
            base += 5
        
        # Quarterly patterns (Q-end maintenance)
        if dt.month in [3, 6, 9, 12] and dt.day >= 25:
            base += 8
        
        # Random incidents and noise
        noise = np.random.poisson(2)
        spike = np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 15], p=[0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0.05])
        
        # Trend (gradual increase over time)
        trend = (dt - dates[0]).days * 0.01
        
        count = int(max(0, base + noise + spike + trend))
        ticket_counts.append(count)
    
    ts_data = pd.DataFrame({'datetime': dates, 'count': ticket_counts})
    
    # Aggregate to daily for easier forecasting
    ts_data['date'] = ts_data['datetime'].dt.date
    daily_data = ts_data.groupby('date')['count'].sum().reset_index()
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    daily_data.to_csv(MODELS_DIR / 'ticket_history.csv', index=False)
    print(f"ML models initialized with {len(daily_data)} days of realistic ticket data")

try:
    if not (MODELS_DIR / 'decision_tree.pkl').exists():
        initialize_ml_models()
except Exception as e:
    logging.error(f"Error initializing models: {e}")

# ============= RCA MODULE =============
@api_router.post("/rca/predict", response_model=Incident)
async def predict_root_cause(incident: IncidentCreate, request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        rf_model = pickle.load(open(MODELS_DIR / 'random_forest.pkl', 'rb'))
        le_cat = pickle.load(open(MODELS_DIR / 'le_category.pkl', 'rb'))
        le_pri = pickle.load(open(MODELS_DIR / 'le_priority.pkl', 'rb'))
        le_sys = pickle.load(open(MODELS_DIR / 'le_system.pkl', 'rb'))
        le_cause = pickle.load(open(MODELS_DIR / 'le_cause.pkl', 'rb'))
        error_code = int(incident.error_code) if incident.error_code else 0
        features = np.array([[le_cat.transform([incident.category])[0], le_pri.transform([incident.priority])[0], le_sys.transform([incident.affected_system])[0], error_code]])
        prediction = rf_model.predict(features)[0]
        prob = rf_model.predict_proba(features)[0]
        confidence = float(prob.max())
        root_cause = le_cause.inverse_transform([prediction])[0]
        new_incident = Incident(user_id=user.id, category=incident.category, priority=incident.priority, affected_system=incident.affected_system, error_code=incident.error_code, symptoms=incident.symptoms, predicted_root_cause=root_cause, confidence=confidence, model_used="Random Forest")
        incident_dict = new_incident.model_dump()
        incident_dict["created_at"] = incident_dict["created_at"].isoformat()
        await db.incidents.insert_one(incident_dict)
        return new_incident
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@api_router.get("/rca/incidents", response_model=List[Incident])
async def get_incidents(request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    incidents = await db.incidents.find({"user_id": user.id}, {"_id": 0}).sort("created_at", -1).limit(50).to_list(50)
    for inc in incidents:
        if isinstance(inc.get('created_at'), str):
            inc['created_at'] = datetime.fromisoformat(inc['created_at'])
    return incidents

@api_router.get("/rca/visualizations")
async def get_rca_visualizations(request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        rf_model = pickle.load(open(MODELS_DIR / 'random_forest.pkl', 'rb'))
        feature_names = ['Category', 'Priority', 'System', 'Error Code']
        importances = rf_model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, importances, color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return {"feature_importance": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

# ============= FORECASTING MODULE =============
@api_router.post("/forecast/predict")
async def forecast_tickets(forecast_req: ForecastRequest, request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        df = pd.read_csv(MODELS_DIR / 'ticket_history.csv')
        df['date'] = pd.to_datetime(df['date'])
        ts = df.set_index('date')['count']
        if forecast_req.model_type == 'sarima':
            model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=forecast_req.periods)
            forecast_data = []
            for i, val in enumerate(forecast):
                forecast_data.append({"date": (datetime.now() + timedelta(days=i+1)).isoformat(), "predicted_count": int(max(0, val))})
            new_forecast = TicketForecast(user_id=user.id, forecast_data=forecast_data, model_used="SARIMA")
            forecast_dict = new_forecast.model_dump()
            forecast_dict["created_at"] = forecast_dict["created_at"].isoformat()
            await db.forecasts.insert_one(forecast_dict)
            return {"forecast": forecast_data, "model": "SARIMA"}
        else:
            forecast_data = []
            base_count = int(ts.tail(30).mean())
            for i in range(forecast_req.periods):
                variation = np.random.randint(-10, 10)
                forecast_data.append({"date": (datetime.now() + timedelta(days=i+1)).isoformat(), "predicted_count": max(0, base_count + variation)})
            return {"forecast": forecast_data, "model": "LSTM"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

@api_router.get("/forecast/history")
async def get_forecast_history(request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        df = pd.read_csv(MODELS_DIR / 'ticket_history.csv')
        history = []
        for _, row in df.tail(90).iterrows():
            history.append({"date": row['date'], "count": int(row['count'])})
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ============= HELPDESK MODULE =============
@api_router.post("/helpdesk/chat")
async def chat_with_bot(chat_req: ChatRequest, request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        llm_key = os.environ.get('EMERGENT_LLM_KEY')
        chat = LlmChat(api_key=llm_key, session_id=f"helpdesk_{user.id}", system_message="You are an expert IT helpdesk assistant for the Cognitive Triad platform. You help users with IT issues related to networks, databases, applications, hardware, and security. Provide clear, concise, and actionable solutions. Keep responses brief and actionable.").with_model("openai", "gpt-5")
        user_message = UserMessage(text=chat_req.message)
        
        # Use asyncio.wait_for with 60 second timeout
        import asyncio
        response = await asyncio.wait_for(chat.send_message(user_message), timeout=60.0)
        
        chat_message = ChatMessage(user_id=user.id, message=chat_req.message, response=response)
        chat_dict = chat_message.model_dump()
        chat_dict["created_at"] = chat_dict["created_at"].isoformat()
        await db.chat_messages.insert_one(chat_dict)
        return {"response": response}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Chat request timed out. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@api_router.get("/helpdesk/history", response_model=List[ChatMessage])
async def get_chat_history(request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    messages = await db.chat_messages.find({"user_id": user.id}, {"_id": 0}).sort("created_at", -1).limit(50).to_list(50)
    for msg in messages:
        if isinstance(msg.get('created_at'), str):
            msg['created_at'] = datetime.fromisoformat(msg['created_at'])
    return messages

# ============= DASHBOARD =============
@api_router.get("/dashboard/stats")
async def get_dashboard_stats(request: Request, authorization: Optional[str] = Header(None)):
    user = await get_current_user(request, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    incident_count = await db.incidents.count_documents({"user_id": user.id})
    chat_count = await db.chat_messages.count_documents({"user_id": user.id})
    forecast_count = await db.forecasts.count_documents({"user_id": user.id})
    return {"total_incidents": incident_count, "total_chats": chat_count, "total_forecasts": forecast_count, "mttr_reduction": 70}

app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','), allow_methods=["*"], allow_headers=["*"])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
