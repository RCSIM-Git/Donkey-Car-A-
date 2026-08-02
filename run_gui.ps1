$ErrorActionPreference = "Stop"
$ScriptFolder = $PSScriptRoot

$PythonExe = Join-Path $ScriptFolder ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$env:PYTHONPATH = "$ScriptFolder;$ScriptFolder\core_engine;d:\DonkeyPROJECT\MonacoRacing;" + $env:PYTHONPATH

Write-Host "========================================================="
Write-Host " STARTING MONACO GP - AUTONOMOUS COMMAND CENTER (GUI) "
Write-Host "========================================================="
& $PythonExe "$ScriptFolder\monaco_dashboard.py"
