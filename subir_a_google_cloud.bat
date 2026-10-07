@echo off
title Subir VeritasAI a Google Cloud Run
color 0A
cls
echo ======================================================================
echo          SUBIR VERITAS AI A GOOGLE CLOUD RUN
echo ======================================================================
echo.
echo 1. Seleccionando archivo 'veritas_detector_cloud.zip' en tu carpeta...
explorer.exe /select,"%~dp0veritas_detector_cloud.zip"
echo.
echo 2. Abriendo Google Cloud Shell en tu navegador con tu proyecto...
start "" "https://shell.cloud.google.com/?project=majestic-carrier-b04qf"
echo.
echo 3. Copiando el comando al portapapeles...
powershell -Command "Set-Clipboard -Value 'unzip -o veritas_detector_cloud.zip -d veritas && cd veritas && gcloud run deploy veritas-detector --source . --region us-central1 --allow-unauthenticated'"
echo.
echo ======================================================================
echo   INSTRUCCIONES SIMPLES:
echo ======================================================================
echo   1. En Cloud Shell (la pantalla negra que se abrio en tu navegador):
echo      - Arrastra el archivo 'veritas_detector_cloud.zip' a esa pantalla
echo        (o dale a los 3 puntitos arriba a la derecha -> 'Upload').
echo.
echo   2. Haz clic dentro de la terminal negra, presiona Ctrl + V
echo      (el comando ya esta copiado en tu portapapeles) y pulsa ENTER.
echo.
echo   3. Si te pide confirmacion (Y/n), escribe Y y pulsa ENTER.
echo ======================================================================
echo.
pause
