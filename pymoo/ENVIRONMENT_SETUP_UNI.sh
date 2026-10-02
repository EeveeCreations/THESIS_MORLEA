@echo off

echo ========================================
echo Creating virtual environment...
echo ========================================

python -m venv RL_MOEA --without-pip

echo ========================================
echo Activating virtual environment...
echo ========================================

source RL_MOEA\bin\activate

echo ========================================
echo  Add Pip
echo ========================================

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

python get-pip.py

echo ========================================
echo  Upgrade pip...
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
