@echo off

REM Get the absolute path of the current directory
set current_dir=%~dp0

REM Construct the full path to the virtual environment activation script
set venv_path="%current_dir%\objdet\Scripts\activate"

REM Activate the virtual environment
call %venv_path%