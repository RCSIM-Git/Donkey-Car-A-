$ErrorActionPreference = "Stop"
$ScriptFolder = $PSScriptRoot

$PythonExe = Join-Path $ScriptFolder ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$env:PYTHONPATH = "$ScriptFolder;$ScriptFolder\core_engine;d:\DonkeyPROJECT\MonacoRacing;" + $env:PYTHONPATH

Write-Host "============================"
Write-Host "1. START MAPPING (SLAM)"
Write-Host "============================"
& $PythonExe "$ScriptFolder\01_MAPPING\run_slam_mapping.py"

Write-Host "============================"
Write-Host "2. GENERATING OPTIMAL RACING LINE (A*)"
Write-Host "============================"
& $PythonExe "$ScriptFolder\02_PATH_PLANNING\generate_optimal_path.py"

Write-Host "============================"
Write-Host "3. A* DRIVE AND TUB RECORDING"
Write-Host "============================"
& $PythonExe "$ScriptFolder\02_PATH_PLANNING\drive_optimal_path_and_collect.py"

Write-Host "============================"
Write-Host "4. EXPERT TRAINING (BEHAVIORAL CLONING)"
Write-Host "============================"
& $PythonExe "$ScriptFolder\03_BC_EXPERT\train_bc_monaco.py"

Write-Host "============================"
Write-Host "5. PPO RL TRAINING (Tuning <21s)"
Write-Host "============================"
& $PythonExe "$ScriptFolder\04_RL_PPO\run_ppo.py"

Write-Host "--- WORKFLOW COMPLETED SUCCESSFULLY ---"
