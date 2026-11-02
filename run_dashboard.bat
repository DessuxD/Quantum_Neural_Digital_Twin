@echo off
echo ======================================================================
echo  QUANTUM NEURAL DIGITAL TWIN - FACULTY DEFENSE DASHBOARD
echo ======================================================================
echo Starting server on http://localhost:5000 ...
set PYTHONPATH=%CD%
python src/web/app.py
pause
