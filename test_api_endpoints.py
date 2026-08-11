#!/usr/bin/env python3
"""
SmartCrypto API Endpoint Tester
Tests all API endpoints to ensure they're working correctly
"""

import requests
import json
import time
import sys
from typing import Dict, List, Optional

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.results = []
        
    def test_endpoint(self, method: str, endpoint: str, name: str, expected_status: int = 200, **kwargs):
        """Test a single endpoint and record results"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            response = getattr(self.session, method.lower())(url, timeout=10, **kwargs)
            response_time = time.time() - start_time
            
            success = response.status_code == expected_status
            status = "✅ PASS" if success else "❌ FAIL"
            
            result = {
                "name": name,
                "endpoint": endpoint,
                "method": method.upper(),
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time": round(response_time, 3),
                "success": success,
                "error": None
            }
            
            # Try to parse JSON response
            try:
                result["response_data"] = response.json()
            except:
                result["response_data"] = response.text[:200]  # First 200 chars
            
            self.results.append(result)
            
            print(f"{status} {method.upper():<6} {endpoint:<40} "
                  f"{response.status_code} (expected {expected_status}) "
                  f"[{response_time:.3f}s]")
                  
            if not success and response.status_code != 404:  # Don't show details for 404s
                print(f"   Response: {result['response_data']}")
                
            return result
            
        except requests.exceptions.RequestException as e:
            error_result = {
                "name": name,
                "endpoint": endpoint,
                "method": method.upper(),
                "status_code": None,
                "expected_status": expected_status,
                "response_time": None,
                "success": False,
                "error": str(e),
                "response_data": None
            }
            self.results.append(error_result)
            print(f"❌ ERROR {method.upper():<6} {endpoint:<40} [Error: {e}]")
            return error_result
    
    def run_all_tests(self):
        """Run comprehensive tests for all endpoints"""
        print(f"🚀 Testing SmartCrypto API at {self.base_url}")
        print("=" * 80)
        
        # Health & Status Endpoints
        print("\n📊 Health & Status Endpoints:")
        print("-" * 40)
        self.test_endpoint("GET", "/", "Root endpoint")
        self.test_endpoint("GET", "/api/v1/health", "Health check")
        self.test_endpoint("GET", "/api/v1/live", "Liveness check")
        self.test_endpoint("GET", "/api/v1/ready", "Readiness check")
        self.test_endpoint("GET", "/api/v1/status", "System status")
        self.test_endpoint("GET", "/system-status", "Detailed system status")
        
        # Signal Endpoints
        print("\n📡 Signal Endpoints:")
        print("-" * 40)
        self.test_endpoint("GET", "/api/v1/signals", "All signals")
        
        # Test signals for specific symbols
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        for symbol in symbols:
            self.test_endpoint("GET", f"/api/v1/signals/{symbol}", f"Signal for {symbol}", expected_status=200)
        
        # Portfolio Endpoints
        print("\n💰 Portfolio Endpoints:")
        print("-" * 40)
        self.test_endpoint("GET", "/api/v1/portfolio/recommendations", "Portfolio recommendations")
        self.test_endpoint("GET", "/api/v1/portfolio/history", "Portfolio history")
        self.test_endpoint("GET", "/api/v1/portfolio/positions", "Portfolio positions")
        self.test_endpoint("GET", "/portfolio/overview", "Portfolio overview")
        self.test_endpoint("GET", "/portfolio/analytics", "Portfolio analytics")
        
        # Position Management Endpoints
        print("\n🎯 Position Management Endpoints:")
        print("-" * 40)
        self.test_endpoint("GET", "/api/v1/positions/active", "Active positions")
        self.test_endpoint("GET", "/api/v1/positions/stats/summary", "Positions summary")
        
        # Test position endpoints for specific symbols (might 404 if no positions)
        for symbol in symbols:
            self.test_endpoint("GET", f"/api/v1/positions/{symbol}", f"Position for {symbol}", expected_status=404)
        
        # History & Performance Endpoints
        print("\n📈 History & Performance Endpoints:")
        print("-" * 40)
        self.test_endpoint("GET", "/api/v1/history/signals", "Signal history")
        self.test_endpoint("GET", "/api/v1/history/patterns", "Pattern performance")
        self.test_endpoint("GET", "/api/v1/performance", "Performance metrics")
        
        # Test symbol-specific history endpoints
        for symbol in symbols:
            self.test_endpoint("GET", f"/api/v1/history/performance/{symbol}", f"Performance for {symbol}")
        
        # Test & Debug Endpoints
        print("\n🔧 Test & Debug Endpoints:")
        print("-" * 40)
        for symbol in symbols:
            self.test_endpoint("GET", f"/api/v1/test/data-quality/{symbol}", f"Data quality for {symbol}")
            self.test_endpoint("GET", f"/api/v1/check/data-quality/{symbol}", f"Check data quality for {symbol}")
            self.test_endpoint("GET", f"/debug/websocket/{symbol}", f"WebSocket debug for {symbol}")
        
        # WebSocket Info Endpoint
        self.test_endpoint("GET", "/api/v1/ws/positions/updates", "WebSocket positions info")
        
        # Test endpoints (might require more setup)
        for symbol in symbols:
            self.test_endpoint("POST", f"/api/v1/test/signal/{symbol}", f"Test signal for {symbol}", expected_status=200)
        
        # Retraining endpoint
        self.test_endpoint("POST", "/api/v1/retrain", "Trigger retraining", expected_status=200)
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        # Calculate average response time
        response_times = [r['response_time'] for r in self.results if r['response_time'] is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"⏱️  Average Response Time: {avg_response_time:.3f}s")
        
        # Show failed tests
        if failed_tests > 0:
            print(f"\n🔍 Failed Tests:")
            for result in self.results:
                if not result['success']:
                    print(f"   ❌ {result['method']} {result['endpoint']}")
                    if result['error']:
                        print(f"      Error: {result['error']}")
                    elif result['status_code']:
                        print(f"      Status: {result['status_code']} (expected {result['expected_status']})")
        
        # Show slow endpoints (> 1 second)
        slow_endpoints = [r for r in self.results if r['response_time'] and r['response_time'] > 1.0]
        if slow_endpoints:
            print(f"\n🐌 Slow Endpoints (> 1s):")
            for result in slow_endpoints:
                print(f"   ⏱️  {result['method']} {result['endpoint']}: {result['response_time']:.3f}s")
        
        # Overall status
        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! Your API is working perfectly! 🎉")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Check the failed endpoints above.")
    
    def save_results(self, filename: str = "api_test_results.json"):
        """Save test results to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_url": self.base_url,
                "results": self.results,
                "summary": {
                    "total_tests": len(self.results),
                    "passed_tests": sum(1 for r in self.results if r['success']),
                    "failed_tests": len(self.results) - sum(1 for r in self.results if r['success'])
                }
            }, f, indent=2)
        print(f"\n💾 Results saved to {filename}")

def main():
    # You can change the base URL here if needed
    base_url = "http://localhost:8000"
    
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    tester = APITester(base_url)
    
    try:
        tester.run_all_tests()
        tester.save_results()
        
        # Exit with error code if any tests failed
        failed_tests = len([r for r in tester.results if not r['success']])
        sys.exit(1 if failed_tests > 0 else 0)
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()