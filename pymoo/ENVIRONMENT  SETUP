@echo off

echo ========================================
echo Creating virtual environment...
echo ========================================

python -m venv RL_MOEA

echo ========================================
echo Activating virtual environment...
echo ========================================

call RL_MOEA\Scripts\activate.bat

echo ========================================
echo Upgrading pip...
echo ========================================

python -m pip install --upgrade pip

echo ========================================
echo Installing requirements...
echo ========================================

pip install -r requirements.txt

echo ========================================
echo Setup complete!
echo ========================================
echo To activate the environment later:
echo RL_MOEA\Scripts\activate

pause
