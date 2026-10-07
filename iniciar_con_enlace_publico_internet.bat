@echo off
title VeritasAI - Servidor Publico en Internet
color 0A
cls
echo ======================================================================
echo       VERITAS AI - INICIANDO ENLACE PUBLICO EN INTERNET
echo ======================================================================
echo.
echo 1. Iniciando servidor web local...
start "VeritasAI Servidor" /min python web_app.py
timeout /t 2 >nul
echo.
echo 2. Conectando tunel publico a internet con Cloudflare...
start "Cloudflare Tunnel" /min "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8000
timeout /t 3 >nul
echo.
echo ======================================================================
echo   ¡EL SERVIDOR ESTA EN VIVO EN INTERNET!
echo   Cualquiera con el enlace puede acceder desde su celular o PC.
echo ======================================================================
echo.
pause
