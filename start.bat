@echo off
echo ======================================
echo  EmoIA - Intelligence Artificielle
echo ======================================

REM Verifier Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

REM Verifier Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

REM Creer environnement virtuel si necessaire
if not exist "venv" (
    echo Creation de l'environnement virtuel...
    python -m venv venv
)

REM Activer environnement virtuel
echo Activation de l'environnement virtuel...
call venv\Scripts\activate

REM Installer dependances Python
echo Installation des dependances Python...
pip install -r requirements.txt

REM Creer les repertoires
echo Creation des repertoires...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "cache" mkdir cache

REM Telecharger ressources NLTK
echo Telechargement des ressources NLTK...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

REM Installer dependances frontend
echo Installation des dependances frontend...
cd frontend
call npm install
cd ..

REM Demarrer les services
echo ======================================
echo Demarrage des services...
echo ======================================

REM Demarrer backend
start "EmoIA Backend" cmd /k "venv\Scripts\activate && python -m uvicorn src.core.api:app --host 0.0.0.0 --port 8000 --reload"

REM Attendre que le backend demarre
timeout /t 5 /nobreak >nul

REM Demarrer frontend
cd frontend
start "EmoIA Frontend" cmd /k "npm start"
cd ..

echo ======================================
echo  EmoIA est maintenant demarre !
echo ======================================
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ======================================
echo.
echo Fermez cette fenetre pour arreter tous les services
pause