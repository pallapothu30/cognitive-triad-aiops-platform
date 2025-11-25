#!/usr/bin/env python3
"""
Cognitive Triad Backend API Testing Suite
Tests all three modules: RCA, Forecasting, and Helpdesk
"""

import requests
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional

class CognitiveTriadTester:
    def __init__(self, base_url="https://cognitive-triad.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session_token = None
        self.user_id = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test_name": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}: {details}")

    def setup_test_user(self):
        """Create test user and session in MongoDB"""
        print("\n🔧 Setting up test user and session...")
        
        try:
            # Create test user and session via MongoDB
            timestamp = int(datetime.now().timestamp())
            user_id = f"test-user-{timestamp}"
            session_token = f"test_session_{timestamp}"
            email = f"test.user.{timestamp}@example.com"
            
            mongo_script = f"""
            use('test_database');
            var userId = '{user_id}';
            var sessionToken = '{session_token}';
            var email = '{email}';
            
            db.users.insertOne({{
                id: userId,
                email: email,
                name: 'Test User',
                picture: 'https://via.placeholder.com/150',
                created_at: new Date()
            }});
            
            db.user_sessions.insertOne({{
                user_id: userId,
                session_token: sessionToken,
                expires_at: new Date(Date.now() + 7*24*60*60*1000),
                created_at: new Date()
            }});
            
            print('SUCCESS: User and session created');
            """
            
            result = subprocess.run(
                ['mongosh', '--eval', mongo_script],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and 'SUCCESS' in result.stdout:
                self.session_token = session_token
                self.user_id = user_id
                self.log_test("Setup Test User", True, f"Created user {user_id}")
                return True
            else:
                self.log_test("Setup Test User", False, f"MongoDB error: {result.stderr}")
                return False
                
        except Exception as e:
            self.log_test("Setup Test User", False, f"Exception: {str(e)}")
            return False

    def make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, expected_status: int = 200) -> tuple:
        """Make authenticated API request"""
        url = f"{self.api_url}/{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.session_token}' if self.session_token else ''
        }
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                return False, f"Unsupported method: {method}"
            
            success = response.status_code == expected_status
            
            if success:
                try:
                    return True, response.json()
                except:
                    return True, response.text
            else:
                return False, f"Status {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            return False, "Request timeout"
        except Exception as e:
            return False, f"Request error: {str(e)}"

    def test_auth_endpoints(self):
        """Test authentication endpoints"""
        print("\n🔐 Testing Authentication Endpoints...")
        
        # Test /auth/me
        success, response = self.make_request('GET', 'auth/me')
        if success and isinstance(response, dict) and 'email' in response:
            self.log_test("GET /auth/me", True, f"User authenticated: {response.get('email')}")
        else:
            self.log_test("GET /auth/me", False, f"Auth failed: {response}")

    def test_dashboard_endpoints(self):
        """Test dashboard statistics"""
        print("\n📊 Testing Dashboard Endpoints...")
        
        success, response = self.make_request('GET', 'dashboard/stats')
        if success and isinstance(response, dict):
            required_fields = ['total_incidents', 'total_chats', 'total_forecasts', 'mttr_reduction']
            if all(field in response for field in required_fields):
                self.log_test("GET /dashboard/stats", True, f"Stats: {response}")
            else:
                self.log_test("GET /dashboard/stats", False, f"Missing fields in response: {response}")
        else:
            self.log_test("GET /dashboard/stats", False, f"Failed: {response}")

    def test_rca_endpoints(self):
        """Test Root Cause Analysis endpoints"""
        print("\n🧠 Testing RCA Endpoints...")
        
        # Test incident prediction
        incident_data = {
            "category": "Network",
            "priority": "High",
            "affected_system": "Web Server",
            "error_code": "500",
            "symptoms": "Server responding slowly and timing out"
        }
        
        success, response = self.make_request('POST', 'rca/predict', incident_data)
        if success and isinstance(response, dict):
            required_fields = ['predicted_root_cause', 'confidence', 'model_used']
            if all(field in response for field in required_fields):
                self.log_test("POST /rca/predict", True, 
                            f"Root cause: {response.get('predicted_root_cause')} "
                            f"(Confidence: {response.get('confidence', 0)*100:.1f}%)")
            else:
                self.log_test("POST /rca/predict", False, f"Missing fields: {response}")
        else:
            self.log_test("POST /rca/predict", False, f"Failed: {response}")
        
        # Test incidents list
        success, response = self.make_request('GET', 'rca/incidents')
        if success and isinstance(response, list):
            self.log_test("GET /rca/incidents", True, f"Retrieved {len(response)} incidents")
        else:
            self.log_test("GET /rca/incidents", False, f"Failed: {response}")
        
        # Test visualizations
        success, response = self.make_request('GET', 'rca/visualizations')
        if success and isinstance(response, dict) and 'feature_importance' in response:
            self.log_test("GET /rca/visualizations", True, "Feature importance chart generated")
        else:
            self.log_test("GET /rca/visualizations", False, f"Failed: {response}")

    def test_forecast_endpoints(self):
        """Test forecasting endpoints"""
        print("\n📈 Testing Forecast Endpoints...")
        
        # Test SARIMA forecast
        forecast_data = {"model_type": "sarima", "periods": 30}
        success, response = self.make_request('POST', 'forecast/predict', forecast_data)
        if success and isinstance(response, dict) and 'forecast' in response:
            forecast_list = response['forecast']
            if isinstance(forecast_list, list) and len(forecast_list) > 0:
                self.log_test("POST /forecast/predict (SARIMA)", True, 
                            f"Generated {len(forecast_list)} day forecast")
            else:
                self.log_test("POST /forecast/predict (SARIMA)", False, "Empty forecast data")
        else:
            self.log_test("POST /forecast/predict (SARIMA)", False, f"Failed: {response}")
        
        # Test LSTM forecast
        forecast_data = {"model_type": "lstm", "periods": 15}
        success, response = self.make_request('POST', 'forecast/predict', forecast_data)
        if success and isinstance(response, dict) and 'forecast' in response:
            self.log_test("POST /forecast/predict (LSTM)", True, "LSTM forecast generated")
        else:
            self.log_test("POST /forecast/predict (LSTM)", False, f"Failed: {response}")
        
        # Test forecast history
        success, response = self.make_request('GET', 'forecast/history')
        if success and isinstance(response, dict) and 'history' in response:
            history = response['history']
            if isinstance(history, list) and len(history) > 0:
                self.log_test("GET /forecast/history", True, f"Retrieved {len(history)} historical records")
            else:
                self.log_test("GET /forecast/history", False, "No historical data")
        else:
            self.log_test("GET /forecast/history", False, f"Failed: {response}")

    def test_helpdesk_endpoints(self):
        """Test AI helpdesk endpoints"""
        print("\n💬 Testing Helpdesk Endpoints...")
        
        # Test chat
        chat_data = {"message": "My server is running slow, what could be the issue?"}
        success, response = self.make_request('POST', 'helpdesk/chat', chat_data)
        if success and isinstance(response, dict) and 'response' in response:
            ai_response = response['response']
            if ai_response and len(ai_response) > 10:  # Reasonable response length
                self.log_test("POST /helpdesk/chat", True, f"AI responded: {ai_response[:100]}...")
            else:
                self.log_test("POST /helpdesk/chat", False, "Empty or too short AI response")
        else:
            self.log_test("POST /helpdesk/chat", False, f"Failed: {response}")
        
        # Test chat history
        success, response = self.make_request('GET', 'helpdesk/history')
        if success and isinstance(response, list):
            self.log_test("GET /helpdesk/history", True, f"Retrieved {len(response)} chat messages")
        else:
            self.log_test("GET /helpdesk/history", False, f"Failed: {response}")

    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n⚠️ Testing Error Handling...")
        
        # Test invalid incident data
        invalid_data = {"category": "Invalid"}
        success, response = self.make_request('POST', 'rca/predict', invalid_data, expected_status=422)
        if not success and "422" in str(response):
            self.log_test("Invalid RCA Data Handling", True, "Properly rejected invalid data")
        else:
            self.log_test("Invalid RCA Data Handling", False, f"Should have failed: {response}")
        
        # Test unauthenticated request
        old_token = self.session_token
        self.session_token = None
        success, response = self.make_request('GET', 'dashboard/stats', expected_status=401)
        self.session_token = old_token
        
        if not success and "401" in str(response):
            self.log_test("Unauthenticated Request Handling", True, "Properly rejected unauthenticated request")
        else:
            self.log_test("Unauthenticated Request Handling", False, f"Should have failed: {response}")

    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Cognitive Triad Backend API Tests")
        print(f"Testing against: {self.base_url}")
        
        if not self.setup_test_user():
            print("❌ Failed to setup test user. Aborting tests.")
            return False
        
        # Run all test suites
        self.test_auth_endpoints()
        self.test_dashboard_endpoints()
        self.test_rca_endpoints()
        self.test_forecast_endpoints()
        self.test_helpdesk_endpoints()
        self.test_error_handling()
        
        # Print summary
        print(f"\n📋 Test Summary:")
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"⚠️ {self.tests_run - self.tests_passed} tests failed")
            return False

def main():
    tester = CognitiveTriadTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())