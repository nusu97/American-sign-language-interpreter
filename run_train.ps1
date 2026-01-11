[CmdletBinding()]
param(
  [Parameter(Mandatory = $false)]
  [int]
  $NumTrainSteps = 10000,

  [Parameter(Mandatory = $false)]
  [string]
  $ModelDir = "Tensorflow/workspace/models/my_ssd_mobnet",

  [Parameter(Mandatory = $false)]
  [string]
  $PipelineConfigPath = "Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config",

  [Parameter(Mandatory = $false)]
  [string]
  $PythonExe = ".venv/Scripts/python.exe",

  [Parameter(Mandatory = $false)]
  [string]
  $LogPath = "train.log",

  [Parameter(Mandatory = $false)]
  [switch]
  $AppendLog,

  [Parameter(Mandatory = $false)]
  [switch]
  $NoWait,

  [Parameter(Mandatory = $false)]
  [switch]
  $KillExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot $PythonExe

if (-not (Test-Path $pythonPath)) {
  throw "Python executable not found: $pythonPath. Activate/create the venv first, or pass -PythonExe."
}

$researchDir = Join-Path $repoRoot "Tensorflow/models/research"
if (-not (Test-Path $researchDir)) {
  throw "TensorFlow models research dir not found: $researchDir"
}

$modelDirPath = (Join-Path $repoRoot $ModelDir)
if (-not (Test-Path $modelDirPath)) {
  New-Item -ItemType Directory -Path $modelDirPath | Out-Null
}
$modelDirAbs = (Resolve-Path $modelDirPath).Path
$pipelineConfigAbs = (Resolve-Path (Join-Path $repoRoot $PipelineConfigPath)).Path

$modelsRoot = (Resolve-Path (Join-Path $researchDir "..")).Path

if ($AppendLog) {
  $baseName = [System.IO.Path]::GetFileNameWithoutExtension($LogPath)
  $extension = [System.IO.Path]::GetExtension($LogPath)
  if (-not $extension) {
    $extension = ".log"
  }
  $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $LogPath = "$baseName`_$timestamp$extension"
}

$logPathAbs = (Join-Path $repoRoot $LogPath)
$logErrPathAbs = "$logPathAbs.err"

$pidFileAbs = Join-Path $modelDirAbs ".train.pid.json"

if (Test-Path $pidFileAbs) {
  try {
    $existing = Get-Content $pidFileAbs -Raw | ConvertFrom-Json
    $existingPid = [int]$existing.Pid
    if ($existingPid -gt 0) {
      $existingProc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
      if ($null -ne $existingProc) {
        if (-not $KillExisting) {
          throw "Training appears to already be running (PID $existingPid). If this is stale, delete $pidFileAbs. If you want to stop it automatically, re-run with -KillExisting."
        }
        Stop-Process -Id $existingPid -Force
      }
    }
  }
  catch {
    if (-not $KillExisting) {
      throw "Found existing pidfile but could not validate/stop it: $pidFileAbs. Delete it if stale, or re-run with -KillExisting. Original error: $($_.Exception.Message)"
    }
  }

  Remove-Item $pidFileAbs -Force -ErrorAction SilentlyContinue
}

$modelDirEscaped = [Regex]::Escape($modelDirAbs)
$existingTrainers = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -match 'model_main_tf2\.py' -and $_.CommandLine -match $modelDirEscaped }

if ($existingTrainers) {
  $pids = ($existingTrainers | Select-Object -ExpandProperty ProcessId) -join ", "
  if (-not $KillExisting) {
    throw "Found an existing training process targeting this model dir (PID(s): $pids). Re-run with -KillExisting to stop it/them."
  }

  foreach ($p in $existingTrainers) {
    Stop-Process -Id $p.ProcessId -Force
  }
}

# Object Detection API expects:
# - research/ and research/slim on PYTHONPATH
# - models/ root on PYTHONPATH so `official` resolves to the repo copy
$env:PYTHONPATH = "$researchDir;$researchDir\slim;$modelsRoot"
$env:PYTHONUNBUFFERED = "1"

Write-Host "Using Python: $pythonPath"
Write-Host "Working dir:  $researchDir"
Write-Host "Model dir:    $modelDirAbs"
Write-Host "Pipeline:     $pipelineConfigAbs"
Write-Host "Train steps:  $NumTrainSteps"
Write-Host "PYTHONPATH:   $env:PYTHONPATH"
Write-Host "Log file:     $logPathAbs"
Write-Host "Log stderr:   $logErrPathAbs"

Push-Location $researchDir
try {
  $arguments = @(
    "-u",
    "object_detection/model_main_tf2.py",
    "--model_dir=$modelDirAbs",
    "--pipeline_config_path=$pipelineConfigAbs",
    "--num_train_steps=$NumTrainSteps",
    "--alsologtostderr"
  )

  $proc = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $researchDir -NoNewWindow -PassThru -RedirectStandardOutput $logPathAbs -RedirectStandardError $logErrPathAbs

  @{
    Pid = $proc.Id
    Python = $pythonPath
    StartedAt = (Get-Date).ToString("o")
    WorkingDirectory = $researchDir
    ModelDir = $modelDirAbs
    PipelineConfig = $pipelineConfigAbs
    NumTrainSteps = $NumTrainSteps
    LogStdout = $logPathAbs
    LogStderr = $logErrPathAbs
  } | ConvertTo-Json | Set-Content -Path $pidFileAbs -Encoding UTF8

  Write-Host "Started training PID: $($proc.Id)"
  if ($NoWait) {
    Write-Host "Detached (-NoWait). Tail logs with: Get-Content -Path \"$logPathAbs\" -Wait and Get-Content -Path \"$logErrPathAbs\" -Wait"
    return
  }

  $proc.WaitForExit()
  $proc.Refresh()
  $exitCode = $proc.ExitCode
  if ($exitCode -ne 0) {
    throw "Training exited with code $exitCode. See log: $logPathAbs"
  }
}
finally {
  if (-not $NoWait) {
    Remove-Item $pidFileAbs -Force -ErrorAction SilentlyContinue
  }
  Pop-Location
}
