@title Autojob v1.0
@echo off

:: Automatically retrieves code from source control and executes via SLURM
:: Written by Xander Neuwirth, January 2020

IF [%1]==[] GOTO NO_ARGUMENT
SET TARGET=%1

:: Black magic to save git branch and remote to %BRANCH% and %REMOTE%
FOR /F "tokens=*" %%g IN ('git rev-parse --abbrev-ref HEAD') do (SET BRANCH=%%g)
FOR /F "tokens=*" %%g IN ('git config --get remote.origin.url') do (SET REMOTE=%%g)

git checkout -b %TARGET%
git add .
git commit -m "%TARGET% autojob"
git push --set-upstream origin %TARGET%

:: Grab datetime and create ID to easily identify run folder
SET DATETIME=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~1,2%%time:~3,2%%time:~6,2%
SET ID=%TARGET%_%DATETIME%

:: Copy and fill in template script
copy scripts\setup_template.sh scripts\temp.sh
powershell -Command "(gc scripts\temp.sh) -replace '{{BRANCH}}', '%TARGET%' | Out-File -encoding ASCII scripts\temp.sh"
powershell -Command "(gc scripts\temp.sh) -replace '{{REMOTE}}', '%REMOTE%' | Out-File -encoding ASCII scripts\temp.sh"
powershell -Command "(gc scripts\temp.sh) -replace '{{ID}}', '%ID%' | Out-File -encoding ASCII scripts\temp.sh"

:: Execute remote component
putty.exe -ssh neuwirtha@dh-ood.hpc.msoe.edu -pw %ROSIE_ACCESS% -m scripts\temp.sh

:: Return to original branch
git checkout %BRANCH%

:: Clean up temp file after putty session ends
del scripts\temp.sh
EXIT 0

:NO_ARGUMENT
echo ERROR: Please provide an autojob name
EXIT /b 1