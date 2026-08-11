#!/usr/bin/env python3
"""
Quick API Tester - Simple version for daily use
"""

import requests
import sys

def quick_test(base_url="http://localhost:8000"):
    """Quick test of essential endpoints"""
    endpoints = [
        ("/", "Root"),
        ("/api/v1/health", "Health"),
        ("/api/v1/ready", "Ready"),
        ("/api/v1/signals", "Signals"),
        ("/api/v1/portfolio/recommendations", "Portfolio Recs"),
        ("/api/v1/positions/active", "Active Positions"),
    ]
    
    print(f"🔍 Quick testing {base_url}")
    print("-" * 50)
    
    all_ok = True
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} {name:<20} {endpoint:<30} {response.status_code}")
            
            if response.status_code != 200:
                all_ok = False
                
        except Exception as e:
            print(f"❌ {name:<20} {endpoint:<30} ERROR: {e}")
            all_ok = False
    
    print("-" * 50)
    if all_ok:
        print("🎉 All essential endpoints are working!")
    else:
        print("⚠️  Some endpoints have issues. Check the logs above.")
    
    return all_ok

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    success = quick_test(base_url)
    sys.exit(0 if success else 1)