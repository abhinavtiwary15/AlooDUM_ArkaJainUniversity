"""Cross-platform one-command launcher for NoctTwin Clinical Platform.

Launches both FastAPI backend (port 8000) and Vite frontend (port 5173).
Handles graceful shutdown on Ctrl+C.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def main():
    print("=" * 70)
    print("  NoctTwin: Clinical Decision Support Platform")
    print("  Starting FastAPI backend and React frontend...")
    print("=" * 70)

    # 1. Start FastAPI backend
    api_cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"]
    api_proc = subprocess.Popen(api_cmd)
    print(f"[API] Started FastAPI server on http://localhost:8000 (PID: {api_proc.pid})")

    # 2. Start Web Frontend
    web_dir = os.path.abspath("web")
    is_windows = sys.platform.startswith("win")
    web_cmd = "npm run preview -- --port 5173" if os.path.exists(os.path.join(web_dir, "dist")) else "npm run dev -- --port 5173"

    web_proc = subprocess.Popen(web_cmd, cwd=web_dir, shell=True)
    print(f"[WEB] Started React/Vite server on http://localhost:5173 (PID: {web_proc.pid})")

    print("\n" + "-" * 70)
    print("  Access Clinical Web Dashboard:  http://localhost:5173")
    print("  Access OpenAPI Interactive Docs: http://localhost:8000/docs")
    print("  Press Ctrl+C to terminate both servers.")
    print("-" * 70 + "\n")

    def signal_handler(sig, frame):
        print("\nShutting down NoctTwin servers gracefully...")
        try:
            if is_windows:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(api_proc.pid)])
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(web_proc.pid)])
            else:
                api_proc.terminate()
                web_proc.terminate()
        except (OSError, subprocess.SubprocessError):
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            time.sleep(1)
            if api_proc.poll() is not None or web_proc.poll() is not None:
                break
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
