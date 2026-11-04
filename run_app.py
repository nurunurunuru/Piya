import subprocess
import sys
import time

def main():
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("[*] Starting Dhaka Building Height Analysis...")
    print("1. Starting FastAPI Backend (http://localhost:5000)...")
    backend_proc = subprocess.Popen([sys.executable, "backend.py"])
    
    time.sleep(2)
    
    print("2. Starting Web UI Frontend (http://localhost:8000)...")
    frontend_proc = subprocess.Popen([sys.executable, "-m", "http.server", "8000", "--directory", "web_ui"])
    
    print("\n[+] Both servers are running!")
    print("   * Backend API:  http://localhost:5000")
    print("   * Web UI:       http://localhost:8000")
    print("\nPress Ctrl+C to stop both servers.")
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    main()
