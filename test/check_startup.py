# test/check_startup.py
"""
Quick script to test server startup and readiness.
"""

import requests
import time
import sys

def wait_for_server(url="http://localhost:8000", timeout=120):
    """Wait for server to be fully ready."""
    print("⏳ Waiting for server to start...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Check startup status
            response = requests.get(f"{url}/startup-status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                
                if status.get("status") == "ready":
                    print("✅ Server is READY!")
                    return True
                else:
                    print(f"⏳ {status.get('message')}")
            
        except requests.exceptions.ConnectionError:
            print("⏳ Server not responding yet...")
        except Exception as e:
            print(f"⚠️ Check failed: {e}")
        
        time.sleep(2)
    
    print("❌ Server startup timeout")
    return False

if __name__ == "__main__":
    if wait_for_server():
        print("🎉 Server is ready for connections!")
        sys.exit(0)
    else:
        print("💥 Server failed to start in time")
        sys.exit(1)