param(
    [int]$TaskCount = 2,
    [string]$RepoRoot = "D:\ProgrammingProjects",
    [string]$SrcRelPath = "3d_asset\src\selected_uids_for_training_1763",
    [string]$DstHostPath = "D:\ProgrammingProjects\datasets\texture_verse_resolution1328\selected_uids_for_training_1763",
    [string]$ImageName = "blender-image-generation",
    [int]$CameraCount = 12,
    [int]$EeveeRenderSamples = 12,
    [double]$WhiteEnvStrength = 1.0,
    [double]$ShadedExposure = 0.3,
    [int]$RandomSeed = 42,
    [object]$UseFastMode = $true,
    [switch]$KeepVolumes,
    [switch]$KeepSplitDirs
)

$ErrorActionPreference = "Stop"

function Assert-DockerReady {
    $null = Get-Command docker -ErrorAction Stop
    docker version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker CLI is installed but daemon is not reachable. Start Docker Desktop and ensure Linux containers are enabled."
    }
}

$useFastModeRaw = "$UseFastMode".Trim().ToLowerInvariant()
switch -Regex ($useFastModeRaw) {
    "^(1|true|t|yes|y|on)$" { $UseFastMode = $true; break }
    "^(0|false|f|no|n|off)$" { $UseFastMode = $false; break }
    default {
        throw "UseFastMode must be one of: true/false, 1/0, yes/no, on/off. Actual: '$UseFastMode'"
    }
}

function To-PosixPath([string]$path) {
    return ($path -replace "\\", "/")
}

function Get-RelativePosixPath([string]$root, [string]$child) {
    $rootFull = [System.IO.Path]::GetFullPath($root)
    $childFull = [System.IO.Path]::GetFullPath($child)

    $rootWithSlash = if ($rootFull.EndsWith("\")) { $rootFull } else { "$rootFull\" }
    $uriRoot = New-Object System.Uri($rootWithSlash)
    $uriChild = New-Object System.Uri($childFull)
    $relativeUri = $uriRoot.MakeRelativeUri($uriChild)
    $relative = [System.Uri]::UnescapeDataString($relativeUri.ToString())
    return To-PosixPath $relative
}

$scriptRelPath = "docker_images/blender_image_generation/generate_training_images.py"
$scriptInContainer = "/data/$scriptRelPath"
$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$splitRoot = Join-Path $RepoRoot ("3d_asset\tmp\parallel_splits_" + $runId)

$srcHostPath = if ([System.IO.Path]::IsPathRooted($SrcRelPath)) { $SrcRelPath } else { Join-Path $RepoRoot $SrcRelPath }

if (-not (Test-Path $srcHostPath)) {
    throw "Source path not found: $srcHostPath"
}

if (-not (Test-Path (Join-Path $RepoRoot $scriptRelPath))) {
    throw "Script not found: $(Join-Path $RepoRoot $scriptRelPath)"
}

Assert-DockerReady

$assetFiles = New-Object System.Collections.Generic.List[object]
$assetDirs = Get-ChildItem -Path $srcHostPath -Directory | Sort-Object Name
foreach ($dir in $assetDirs) {
    $glbFiles = Get-ChildItem -Path $dir.FullName -Filter *.glb -File
    foreach ($file in $glbFiles) {
        $assetFiles.Add($file)
    }
}

if ($assetFiles.Count -eq 0) {
    throw "No .glb files found under: $srcHostPath"
}

if ($TaskCount -lt 1) {
    throw "TaskCount must be >= 1"
}
if ($TaskCount -gt $assetFiles.Count) {
    $TaskCount = $assetFiles.Count
}

Write-Host "Run ID: $runId"
Write-Host "Assets found: $($assetFiles.Count)"
Write-Host "TaskCount: $TaskCount"

$buckets = @()
for ($i = 0; $i -lt $TaskCount; $i++) {
    $buckets += ,(New-Object System.Collections.Generic.List[object])
}

for ($i = 0; $i -lt $assetFiles.Count; $i++) {
    $bucketIndex = $i % $TaskCount
    $buckets[$bucketIndex].Add($assetFiles[$i])
}

New-Item -ItemType Directory -Force -Path $splitRoot | Out-Null
New-Item -ItemType Directory -Force -Path $DstHostPath | Out-Null

$jobs = New-Object System.Collections.Generic.List[object]

for ($i = 0; $i -lt $TaskCount; $i++) {
    $jobName = ("job_{0:D2}" -f $i)
    $jobSrcHostPath = Join-Path $splitRoot $jobName
    New-Item -ItemType Directory -Force -Path $jobSrcHostPath | Out-Null

    foreach ($asset in $buckets[$i]) {
        $assetFolderName = Split-Path $asset.Directory.FullName -Leaf
        $jobAssetFolder = Join-Path $jobSrcHostPath $assetFolderName
        New-Item -ItemType Directory -Force -Path $jobAssetFolder | Out-Null

        $destFile = Join-Path $jobAssetFolder $asset.Name
        try {
            New-Item -ItemType HardLink -Path $destFile -Target $asset.FullName | Out-Null
        }
        catch {
            Copy-Item -Path $asset.FullName -Destination $destFile -Force
        }
    }

    $jobSrcRel = Get-RelativePosixPath -root $RepoRoot -child $jobSrcHostPath
    $srcInContainer = "/data/$jobSrcRel"

    $containerName = "blender_gen_${runId}_${jobName}"
    $outVolumeName = "blender_out_${runId}_${jobName}"
    $tmpVolumeName = "blender_tmp_${runId}_${jobName}"

    $dockerArgs = @(
        "run", "-d",
        "--gpus", "all",
        "--name", $containerName,
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
        "-e", "LIBGL_ALWAYS_SOFTWARE=0",
        "-v", "${RepoRoot}:/data:ro",
        "-v", "${outVolumeName}:/work/out",
        "-v", "${tmpVolumeName}:/work/tmp",
        $ImageName,
        "-P", $scriptInContainer, "--",
        "--src_folderpath", $srcInContainer,
        "--dst_folderpath", "/work/out/result",
        "--tmp_folderpath", "/work/tmp/${jobName}_debug",
        "--need_occupation", "--need_basecolor", "--need_metallic", "--need_roughness",
        "--need_normal", "--normal_png_only", "--need_view", "--need_shaded",
        "--need_camera_meta", "--need_pointcloud",
        "--use_white_envlight",
        "--white_env_strength", [string]$WhiteEnvStrength,
        "--shaded_exposure", [string]$ShadedExposure,
        "--random_seed", [string]$RandomSeed,
        "--camera_count", [string]$CameraCount,
        "--eevee_render_samples", [string]$EeveeRenderSamples
    )

    if ($UseFastMode) {
        $dockerArgs += "--fast_mode"
    }

    $startedContainerId = docker @dockerArgs
    if ($LASTEXITCODE -ne 0 -or -not $startedContainerId) {
        throw "Failed to start container '$containerName'. Check Docker Desktop status and image '$ImageName'."
    }
    $startedContainerId = $startedContainerId.Trim()
    Write-Host "Started $containerName with $($buckets[$i].Count) assets, container id $startedContainerId"

    $jobs.Add([PSCustomObject]@{
        JobName = $jobName
        ContainerName = $containerName
        OutVolume = $outVolumeName
        TmpVolume = $tmpVolumeName
        AssetCount = $buckets[$i].Count
        ExitCode = $null
    })
}

Write-Host "All containers started. Waiting for completion..."

foreach ($job in $jobs) {
    $exitCodeText = (docker wait $job.ContainerName).Trim()
    $job.ExitCode = [int]$exitCodeText
    Write-Host "$($job.ContainerName) finished with exit code $($job.ExitCode)"

    if ($job.ExitCode -ne 0) {
        Write-Host "Last logs of $($job.ContainerName):"
        docker logs --tail 120 $job.ContainerName
    }
}

Write-Host "Copying outputs from volumes back to host path: $DstHostPath"

foreach ($job in $jobs) {
    if ($job.ExitCode -eq 0) {
        docker run --rm `
            -v "$($job.OutVolume):/from:ro" `
            -v "${DstHostPath}:/to" `
            alpine sh -c "cp -a /from/result/. /to/"
        Write-Host "Copied output of $($job.JobName)"
    }
    else {
        Write-Host "Skipped copy for $($job.JobName) because container failed"
    }
}

Write-Host "Cleanup containers..."
foreach ($job in $jobs) {
    docker rm $job.ContainerName | Out-Null
}

if (-not $KeepVolumes) {
    Write-Host "Cleanup volumes..."
    foreach ($job in $jobs) {
        docker volume rm $job.OutVolume | Out-Null
        docker volume rm $job.TmpVolume | Out-Null
    }
}

if (-not $KeepSplitDirs) {
    Remove-Item -Path $splitRoot -Recurse -Force
}

$success = ($jobs | Where-Object { $_.ExitCode -eq 0 }).Count
$failed = ($jobs | Where-Object { $_.ExitCode -ne 0 }).Count

Write-Host "Done. Success jobs: $success, Failed jobs: $failed"
Write-Host "Output path: $DstHostPath"
