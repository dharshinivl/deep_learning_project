@echo off
echo Starting Frontend Server...
cd frontend
start http://localhost:3000
python -m http.server 3000



