@echo off
echo ========================================
echo FitVision - Deploy to GitHub Pages
echo ========================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo Error: Git not initialized!
    echo Run: git init
    pause
    exit /b 1
)

REM Save current branch
for /f "tokens=*" %%i in ('git branch --show-current') do set CURRENT_BRANCH=%%i
echo Current branch: %CURRENT_BRANCH%
echo.

REM Ask for confirmation
echo This will deploy web/ folder to GitHub Pages
set /p CONFIRM="Continue? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo Step 1: Switching to gh-pages branch...
git checkout gh-pages 2>nul
if errorlevel 1 (
    echo Creating new gh-pages branch...
    git checkout --orphan gh-pages
)

echo.
echo Step 2: Cleaning old files...
git rm -rf . 2>nul

echo.
echo Step 3: Copying web files...
xcopy /E /I /Y web\* . >nul

echo.
echo Step 4: Committing changes...
git add .
git commit -m "Deploy: %date% %time%"

echo.
echo Step 5: Pushing to GitHub...
git push origin gh-pages
if errorlevel 1 (
    echo.
    echo First time push? Run:
    echo git push -u origin gh-pages
    pause
)

echo.
echo Step 6: Returning to %CURRENT_BRANCH%...
git checkout %CURRENT_BRANCH%

echo.
echo ========================================
echo ✅ Deployment Complete!
echo ========================================
echo.
echo Your site will be available at:
echo https://YOUR_USERNAME.github.io/fitvision-app/
echo.
echo Note: Replace YOUR_USERNAME with your GitHub username
echo It may take 1-2 minutes to go live.
echo.
pause
