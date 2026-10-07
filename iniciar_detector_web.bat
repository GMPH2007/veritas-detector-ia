@echo off
title VeritasAI - Detector de Inteligencia Artificial
color 0B
echo ======================================================================
echo           VERITAS AI - DETECTOR AVANZADO DE TEXTO IA
echo    (ChatGPT, Google Gemini, Claude, Perplexity y Humanos)
echo ======================================================================
echo.
echo  Iniciando servidor web local y abriendo interfaz en el navegador...
echo.

start "" "http://127.0.0.1:8000"
python web_app.py

pause
