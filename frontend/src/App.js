import { useEffect, useState } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from "react-router-dom";
import axios from "axios";
import { Toaster, toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Activity, Brain, MessageSquare, BarChart3, LogOut, TrendingUp, AlertCircle } from "lucide-react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const AUTH_URL = "https://auth.emergentagent.com";

// Auth Context
const useAuth = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const response = await axios.get(`${API}/auth/me`, { withCredentials: true });
      setUser(response.data);
    } catch (error) {
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      await axios.post(`${API}/auth/logout`, {}, { withCredentials: true });
      setUser(null);
      window.location.href = "/";
    } catch (error) {
      console.error("Logout error:", error);
    }
  };

  return { user, loading, checkAuth, logout };
};

// Landing Page
const LandingPage = () => {
  const navigate = useNavigate();
  const [showError, setShowError] = useState(false);

  useEffect(() => {
    // Check for auth errors in URL
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('error') === 'auth_failed') {
      setShowError(true);
      // Clean URL
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, []);

  const handleLogin = () => {
    const redirectUrl = encodeURIComponent(`${window.location.origin}/dashboard`);
    window.location.href = `${AUTH_URL}/?redirect=${redirectUrl}`;
  };

  return (
    <div data-testid="landing-page" className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      <div className="container mx-auto px-4 py-12">
        {showError && (
          <div className="max-w-2xl mx-auto mb-8 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-red-900 font-semibold">Authentication Failed</p>
              <p className="text-red-700 text-sm mt-1">
                The login process was interrupted. Please try again. If the issue persists, clear your browser cookies and try again.
              </p>
            </div>
          </div>
        )}
        
        <div className="text-center mb-16">
          <h1 data-testid="app-title" className="text-6xl font-bold text-gray-900 mb-4" style={{fontFamily: 'Playfair Display, serif'}}>
            Cognitive Triad
          </h1>
          <p data-testid="app-subtitle" className="text-xl text-gray-600" style={{fontFamily: 'Inter, sans-serif'}}>
            AI-Powered IT Operational Excellence Platform
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-12">
          <Card data-testid="feature-rca" className="border-2 border-blue-100 hover:border-blue-300 transition-all duration-300 hover:shadow-xl">
            <CardHeader>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <Brain className="w-6 h-6 text-blue-600" />
              </div>
              <CardTitle className="text-2xl">Root Cause Analysis</CardTitle>
              <CardDescription>
                AI-powered RCA using Decision Tree, Random Forest, and Neural Networks to predict incident root causes and reduce MTTR by 70%
              </CardDescription>
            </CardHeader>
          </Card>

          <Card data-testid="feature-forecast" className="border-2 border-cyan-100 hover:border-cyan-300 transition-all duration-300 hover:shadow-xl">
            <CardHeader>
              <div className="w-12 h-12 bg-cyan-100 rounded-lg flex items-center justify-center mb-4">
                <TrendingUp className="w-6 h-6 text-cyan-600" />
              </div>
              <CardTitle className="text-2xl">Load Prediction</CardTitle>
              <CardDescription>
                Time-series forecasting with SARIMA and LSTM models to predict service request volumes with 85%+ accuracy
              </CardDescription>
            </CardHeader>
          </Card>

          <Card data-testid="feature-helpdesk" className="border-2 border-teal-100 hover:border-teal-300 transition-all duration-300 hover:shadow-xl">
            <CardHeader>
              <div className="w-12 h-12 bg-teal-100 rounded-lg flex items-center justify-center mb-4">
                <MessageSquare className="w-6 h-6 text-teal-600" />
              </div>
              <CardTitle className="text-2xl">AI Self-Service</CardTitle>
              <CardDescription>
                GPT-5 powered intelligent helpdesk that automates 50% of common support queries with instant resolutions
              </CardDescription>
            </CardHeader>
          </Card>
        </div>

        <div className="text-center">
          <Button 
            data-testid="login-button"
            onClick={handleLogin} 
            size="lg" 
            className="text-lg px-8 py-6 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
          >
            Get Started with Google
          </Button>
        </div>
      </div>
    </div>
  );
};

// Dashboard Component
const Dashboard = () => {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState("rca");
  const [stats, setStats] = useState(null);

  // RCA State
  const [incident, setIncident] = useState({category: '', priority: '', affected_system: '', error_code: '', symptoms: ''});
  const [rcaResult, setRcaResult] = useState(null);
  const [rcaLoading, setRcaLoading] = useState(false);
  const [incidents, setIncidents] = useState([]);

  // Forecast State
  const [forecastType, setForecastType] = useState('sarima');
  const [forecastPeriods, setForecastPeriods] = useState(30);
  const [forecastData, setForecastData] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [historyData, setHistoryData] = useState([]);

  // Helpdesk State
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    if (user) {
      loadStats();
      loadIncidents();
      loadChatHistory();
      loadForecastHistory();
    }
  }, [user]);

  const loadStats = async () => {
    try {
      const response = await axios.get(`${API}/dashboard/stats`, { withCredentials: true });
      setStats(response.data);
    } catch (error) {
      console.error("Error loading stats:", error);
    }
  };

  const loadIncidents = async () => {
    try {
      const response = await axios.get(`${API}/rca/incidents`, { withCredentials: true });
      setIncidents(response.data);
    } catch (error) {
      console.error("Error loading incidents:", error);
    }
  };

  const loadChatHistory = async () => {
    try {
      const response = await axios.get(`${API}/helpdesk/history`, { withCredentials: true });
      setChatHistory(response.data);
    } catch (error) {
      console.error("Error loading chat history:", error);
    }
  };

  const loadForecastHistory = async () => {
    try {
      const response = await axios.get(`${API}/forecast/history`, { withCredentials: true });
      setHistoryData(response.data.history);
    } catch (error) {
      console.error("Error loading forecast history:", error);
    }
  };

  const handleRCAPredict = async () => {
    if (!incident.category || !incident.priority || !incident.affected_system || !incident.symptoms) {
      toast.error("Please fill all required fields");
      return;
    }
    setRcaLoading(true);
    try {
      const response = await axios.post(`${API}/rca/predict`, incident, { withCredentials: true });
      setRcaResult(response.data);
      toast.success("Root cause predicted successfully!");
      await loadIncidents();
      await loadStats(); // Refresh dashboard stats
      setIncident({category: '', priority: '', affected_system: '', error_code: '', symptoms: ''});
    } catch (error) {
      toast.error("Failed to predict root cause");
      console.error(error);
    } finally {
      setRcaLoading(false);
    }
  };

  const handleForecast = async () => {
    setForecastLoading(true);
    try {
      const response = await axios.post(`${API}/forecast/predict`, {
        model_type: forecastType,
        periods: forecastPeriods
      }, { withCredentials: true });
      setForecastData(response.data.forecast);
      toast.success(`Forecast generated using ${response.data.model}`);
      await loadStats(); // Refresh dashboard stats
    } catch (error) {
      toast.error("Failed to generate forecast");
      console.error(error);
    } finally {
      setForecastLoading(false);
    }
  };

  const handleChat = async () => {
    if (!chatMessage.trim()) return;
    setChatLoading(true);
    const userMsg = chatMessage;
    setChatMessage('');
    try {
      const response = await axios.post(`${API}/helpdesk/chat`, { message: userMsg }, { withCredentials: true });
      setChatHistory([{message: userMsg, response: response.data.response, created_at: new Date().toISOString()}, ...chatHistory]);
      await loadStats(); // Refresh dashboard stats
    } catch (error) {
      toast.error("Failed to send message");
      console.error(error);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div data-testid="dashboard-page" className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      <nav className="bg-white border-b shadow-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 data-testid="dashboard-title" className="text-3xl font-bold text-gray-900" style={{fontFamily: 'Playfair Display, serif'}}>
            Cognitive Triad
          </h1>
          <div className="flex items-center gap-4">
            <span data-testid="user-email" className="text-sm text-gray-600">{user?.email}</span>
            <Button data-testid="logout-button" onClick={logout} variant="outline" size="sm">
              <LogOut className="w-4 h-4 mr-2" /> Logout
            </Button>
          </div>
        </div>
      </nav>

      <div className="container mx-auto px-4 py-8">
        {stats && (
          <div>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-gray-800">Dashboard Overview</h2>
              <Button 
                data-testid="refresh-stats-button" 
                onClick={loadStats} 
                variant="outline" 
                size="sm"
                className="flex items-center gap-2"
              >
                <Activity className="w-4 h-4" /> Refresh Stats
              </Button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <Card data-testid="stat-incidents">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-gray-600">Total Incidents</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-blue-600">{stats.total_incidents}</div>
                </CardContent>
              </Card>
              <Card data-testid="stat-chats">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-gray-600">Chat Interactions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-cyan-600">{stats.total_chats}</div>
                </CardContent>
              </Card>
              <Card data-testid="stat-forecasts">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-gray-600">Forecasts Generated</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-teal-600">{stats.total_forecasts}</div>
                </CardContent>
              </Card>
              <Card data-testid="stat-mttr">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium text-gray-600">MTTR Reduction</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-green-600">{stats.mttr_reduction}%</div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        <Tabs data-testid="main-tabs" value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-8">
            <TabsTrigger data-testid="tab-rca" value="rca" className="text-lg">
              <Brain className="w-5 h-5 mr-2" /> Root Cause Analysis
            </TabsTrigger>
            <TabsTrigger data-testid="tab-forecast" value="forecast" className="text-lg">
              <BarChart3 className="w-5 h-5 mr-2" /> Load Prediction
            </TabsTrigger>
            <TabsTrigger data-testid="tab-helpdesk" value="helpdesk" className="text-lg">
              <MessageSquare className="w-5 h-5 mr-2" /> AI Helpdesk
            </TabsTrigger>
          </TabsList>

          <TabsContent data-testid="rca-content" value="rca">
            <div className="grid md:grid-cols-2 gap-8">
              <Card>
                <CardHeader>
                  <CardTitle>Report New Incident</CardTitle>
                  <CardDescription>Enter incident details to predict root cause</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>Category</Label>
                    <Select data-testid="rca-category" value={incident.category} onValueChange={(v) => setIncident({...incident, category: v})}>
                      <SelectTrigger><SelectValue placeholder="Select category" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Network">Network</SelectItem>
                        <SelectItem value="Database">Database</SelectItem>
                        <SelectItem value="Application">Application</SelectItem>
                        <SelectItem value="Hardware">Hardware</SelectItem>
                        <SelectItem value="Security">Security</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Priority</Label>
                    <Select data-testid="rca-priority" value={incident.priority} onValueChange={(v) => setIncident({...incident, priority: v})}>
                      <SelectTrigger><SelectValue placeholder="Select priority" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Low">Low</SelectItem>
                        <SelectItem value="Medium">Medium</SelectItem>
                        <SelectItem value="High">High</SelectItem>
                        <SelectItem value="Critical">Critical</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Affected System</Label>
                    <Select data-testid="rca-system" value={incident.affected_system} onValueChange={(v) => setIncident({...incident, affected_system: v})}>
                      <SelectTrigger><SelectValue placeholder="Select system" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Web Server">Web Server</SelectItem>
                        <SelectItem value="Database Server">Database Server</SelectItem>
                        <SelectItem value="API Gateway">API Gateway</SelectItem>
                        <SelectItem value="Load Balancer">Load Balancer</SelectItem>
                        <SelectItem value="Storage">Storage</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Error Code (Optional)</Label>
                    <Input data-testid="rca-error-code" value={incident.error_code} onChange={(e) => setIncident({...incident, error_code: e.target.value})} placeholder="e.g., 500" />
                  </div>
                  <div>
                    <Label>Symptoms</Label>
                    <Textarea data-testid="rca-symptoms" value={incident.symptoms} onChange={(e) => setIncident({...incident, symptoms: e.target.value})} placeholder="Describe the issue..." rows={4} />
                  </div>
                  <Button data-testid="rca-predict-button" onClick={handleRCAPredict} disabled={rcaLoading} className="w-full">
                    {rcaLoading ? 'Analyzing...' : 'Predict Root Cause'}
                  </Button>
                  {rcaResult && (
                    <div data-testid="rca-result" className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                      <h3 className="font-bold text-lg mb-2">Prediction Result</h3>
                      <p className="text-gray-700"><strong>Root Cause:</strong> {rcaResult.predicted_root_cause}</p>
                      <p className="text-gray-700"><strong>Confidence:</strong> {(rcaResult.confidence * 100).toFixed(1)}%</p>
                      <p className="text-gray-700"><strong>Model:</strong> {rcaResult.model_used}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Recent Incidents</CardTitle>
                  <CardDescription>Latest analyzed incidents</CardDescription>
                </CardHeader>
                <CardContent>
                  <div data-testid="incidents-list" className="space-y-3 max-h-[600px] overflow-y-auto">
                    {incidents.map((inc) => (
                      <div key={inc.id} className="p-3 border rounded-lg bg-gray-50">
                        <div className="flex justify-between items-start mb-2">
                          <span className="font-semibold text-sm">{inc.category}</span>
                          <span className={`text-xs px-2 py-1 rounded ${inc.priority === 'Critical' ? 'bg-red-100 text-red-700' : inc.priority === 'High' ? 'bg-orange-100 text-orange-700' : 'bg-blue-100 text-blue-700'}`}>{inc.priority}</span>
                        </div>
                        <p className="text-sm text-gray-600 mb-1">{inc.affected_system}</p>
                        <p className="text-sm text-gray-800 font-medium">Root Cause: {inc.predicted_root_cause}</p>
                        <p className="text-xs text-gray-500">Confidence: {(inc.confidence * 100).toFixed(1)}%</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent data-testid="forecast-content" value="forecast">
            <div className="grid md:grid-cols-3 gap-8">
              <Card className="md:col-span-1">
                <CardHeader>
                  <CardTitle>Forecast Configuration</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label>Model Type</Label>
                    <Select data-testid="forecast-model" value={forecastType} onValueChange={setForecastType}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="sarima">SARIMA</SelectItem>
                        <SelectItem value="lstm">LSTM</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Forecast Periods (Days)</Label>
                    <Input data-testid="forecast-periods" type="number" value={forecastPeriods} onChange={(e) => setForecastPeriods(parseInt(e.target.value))} min="7" max="90" />
                  </div>
                  <Button data-testid="forecast-button" onClick={handleForecast} disabled={forecastLoading} className="w-full">
                    {forecastLoading ? 'Generating...' : 'Generate Forecast'}
                  </Button>
                </CardContent>
              </Card>
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>Service Request Volume Forecast</CardTitle>
                </CardHeader>
                <CardContent>
                  {forecastData ? (
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={forecastData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" tick={{fontSize: 12}} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="predicted_count" stroke="#0ea5e9" strokeWidth={2} name="Predicted Tickets" />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : historyData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={historyData.slice(-30)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" tick={{fontSize: 12}} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="count" stroke="#14b8a6" strokeWidth={2} name="Historical Tickets" />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="flex items-center justify-center h-64 text-gray-500">
                      Generate a forecast to see predictions
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent data-testid="helpdesk-content" value="helpdesk">
            <div className="grid md:grid-cols-2 gap-8">
              <Card>
                <CardHeader>
                  <CardTitle>Ask AI Assistant</CardTitle>
                  <CardDescription>Get instant help with IT issues</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <Textarea 
                      data-testid="chat-input"
                      value={chatMessage} 
                      onChange={(e) => setChatMessage(e.target.value)} 
                      placeholder="Describe your IT issue..." 
                      rows={6}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleChat();
                        }
                      }}
                    />
                    <Button data-testid="chat-send-button" onClick={handleChat} disabled={chatLoading} className="w-full">
                      {chatLoading ? 'Sending...' : 'Send Message'}
                    </Button>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Chat History</CardTitle>
                </CardHeader>
                <CardContent>
                  <div data-testid="chat-history" className="space-y-4 max-h-[500px] overflow-y-auto">
                    {chatHistory.map((chat, idx) => (
                      <div key={chat.id || idx} className="space-y-2">
                        <div className="bg-blue-50 p-3 rounded-lg">
                          <p className="text-sm font-semibold text-blue-900">You:</p>
                          <p className="text-sm text-gray-700">{chat.message}</p>
                        </div>
                        <div className="bg-teal-50 p-3 rounded-lg">
                          <p className="text-sm font-semibold text-teal-900">AI Assistant:</p>
                          <p className="text-sm text-gray-700">{chat.response}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

// Session Handler
const SessionHandler = () => {
  const navigate = useNavigate();
  const [processing, setProcessing] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    handleAuthCallback();
  }, []);

  const handleAuthCallback = async () => {
    try {
      // Check for session_id in hash
      const hash = window.location.hash;
      
      if (hash && hash.includes('session_id=')) {
        const sessionId = hash.split('session_id=')[1].split('&')[0];
        
        if (sessionId) {
          await axios.get(`${API}/auth/session`, {
            headers: { 'X-Session-ID': sessionId },
            withCredentials: true
          });
          
          // Clean up URL
          window.history.replaceState(null, '', window.location.pathname);
          
          // Redirect to dashboard
          window.location.href = '/dashboard';
        } else {
          throw new Error('No session ID found');
        }
      } else {
        // No session_id, redirect to home
        navigate('/?error=auth_failed');
      }
    } catch (error) {
      console.error('Auth error:', error);
      const errorMsg = error.response?.data?.detail || error.message || 'Authentication failed';
      setError(errorMsg);
      setTimeout(() => navigate('/?error=auth_failed'), 3000);
    }
  };

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-600" />
          <p className="text-gray-800 font-semibold mb-2">Authentication Error</p>
          <p className="text-gray-600 text-sm">{error}</p>
          <p className="text-gray-500 text-xs mt-2">Redirecting to home...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <Activity className="w-12 h-12 animate-spin mx-auto mb-4 text-blue-600" />
        <p className="text-gray-600">Authenticating...</p>
      </div>
    </div>
  );
};

// Protected Route
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const hash = window.location.hash;
    if (hash && hash.includes('session_id=')) {
      // Don't check auth yet, SessionHandler will process this
      setChecking(false);
      return;
    }
    setChecking(false);
  }, []);

  if (loading || checking) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Activity className="w-12 h-12 animate-spin text-blue-600" />
      </div>
    );
  }

  // Check if we're on the callback URL with session_id
  const hash = window.location.hash;
  if (hash && hash.includes('session_id=')) {
    return <SessionHandler />;
  }

  if (!user) {
    return <Navigate to="/" replace />;
  }

  return children;
};

function App() {
  const [authCallback, setAuthCallback] = useState(false);

  useEffect(() => {
    // Check if this is an auth callback
    const hash = window.location.hash;
    if (hash && hash.includes('session_id=')) {
      setAuthCallback(true);
    }
  }, []);

  return (
    <div className="App">
      <Toaster position="top-right" />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/dashboard" element={
            authCallback ? <SessionHandler /> : 
            <ProtectedRoute><Dashboard /></ProtectedRoute>
          } />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
