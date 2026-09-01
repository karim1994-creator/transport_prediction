@echo off

echo ============================================================
echo TRANSPORT PREDICTION - ML LIFECYCLE
echo ============================================================

cd /d "C:\Users\karim\OneDrive\Documents\Projet Fin etude\transport_prediction_idf\BLOC 5\Transport_prediction"

echo.
echo [1/1] Execution de la pipeline M1/M2
echo.

"C:\Users\karim\anaconda3\python.exe" ml_lifecycle_m1_m2.py

echo.
echo ============================================================
echo PIPELINE TERMINEE
echo ============================================================

exit /b %ERRORLEVEL%