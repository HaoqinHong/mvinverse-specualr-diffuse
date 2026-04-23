param(
  [string]$ConfigName = "textureverse_pi3x_docker",
  [string[]]$ConfigOverrides = @(),
  [string]$ModelPath = "E:\hqhong\ProgrammingProjects\models\Pi3X",
  [string]$ImageName = "da3-3dat:latest",
  [string]$RunsPath = "E:\hqhong\ProgrammingProjects\runs"
)

$ErrorActionPreference = "Stop"

$codePath = "F:\hqhong\ProgrammingProjects\mvinverse"
$datasetPath = "E:\hqhong\ProgrammingProjects\datasets\texture_verse_resolution1024"
$modelPath = $ModelPath
$runsPath = $RunsPath
$imageName = $ImageName

$configPath = Join-Path $codePath "training\config\$ConfigName.yaml"
if (-not (Test-Path $configPath)) {
  throw "Config file not found: $configPath"
}

if (-not (Test-Path $runsPath)) {
  New-Item -ItemType Directory -Force -Path $runsPath | Out-Null
}

function ConvertTo-BashSingleQuotedArg {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Value
  )

  $escapedValue = $Value -replace "'", "'""'""'"
  return "'" + $escapedValue + "'"
}

$configNameArg = ConvertTo-BashSingleQuotedArg $ConfigName
$configOverridesArgs = @()
foreach ($override in $ConfigOverrides) {
  if (-not [string]::IsNullOrWhiteSpace($override)) {
    $configOverridesArgs += ConvertTo-BashSingleQuotedArg $override
  }
}
$configOverridesSuffix = ""
if ($configOverridesArgs.Count -gt 0) {
  $configOverridesSuffix = " " + ($configOverridesArgs -join " ")
}

Write-Host "Training config: $ConfigName"
Write-Host "Container image: $imageName"
Write-Host "Host runs path: $runsPath"
Write-Host "Container runs path: /workspace/runs"
Write-Host "Expected host output root: <RunsPath>/<exp_name from config or overrides>"

docker run --rm --gpus all `
  -v "${codePath}:/workspace/code/mvinverse" `
  -v "${datasetPath}:/workspace/datasets/texture_verse_resolution1024" `
  -v "${modelPath}:/workspace/models/Pi3X" `
  -v "${runsPath}:/workspace/runs" `
  $imageName bash -lc "
    set -e
    cd /workspace/code/mvinverse
    cd training
    torchrun --nproc_per_node=1 launch.py --config $configNameArg$configOverridesSuffix
  "
