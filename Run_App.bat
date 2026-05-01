@echo off
title Student Retention Predictor
echo ================================================
echo   Student Performance ^& Retention Predictor
echo   FABUNMI, Kazeem Olaiya ^| Ph.D. Research
echo   University of Ilorin, Nigeria
echo ================================================
echo.
echo  Starting app... please wait.
echo  Your browser will open automatically.
echo.
echo  To stop the app, close this window.
echo ================================================

cd /d "%~dp0"
py -3.11 -m streamlit run app.py

pause
