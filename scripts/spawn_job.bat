@title Autojob v1.0
@echo off

:: Automatically retrieves code from source control and executes via SLURM
:: Written by Xander Neuwirth, January 2020


:: Black magic to save git branch and remote to %BRANCH% and %REMOTE%
FOR /F "tokens=*" %%g IN ('git rev-parse --abbrev-ref HEAD') do (SET BRANCH=%%g)
FOR /F "tokens=*" %%g IN ('git config --get remote.origin.url') do (SET REMOTE=%%g)

:: Grab datetime and create ID to easily identify run folder
SET DATETIME=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~1,2%%time:~3,2%%time:~6,2%
SET ID=%BRANCH%_%DATETIME%

:: Copy and fill in template script
copy scripts\setup_template.sh scripts\temp.sh
powershell -Command "(gc scripts\temp.sh) -replace '{{BRANCH}}', '%BRANCH%' | Out-File -encoding ASCII scripts\temp.sh"
powershell -Command "(gc scripts\temp.sh) -replace '{{REMOTE}}', '%REMOTE%' | Out-File -encoding ASCII scripts\temp.sh"
powershell -Command "(gc scripts\temp.sh) -replace '{{ID}}', '%ID%' | Out-File -encoding ASCII scripts\temp.sh"

:: Execute remote component
putty.exe -ssh neuwirtha@dh-ood.hpc.msoe.edu -pw %ROSIE_ACCESS% -m scripts\temp.sh

:: Clean up temp file after putty session ends
:: del scripts\temp.sh