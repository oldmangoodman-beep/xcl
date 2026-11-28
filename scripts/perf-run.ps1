# Reproducible paired runtime measurement: an XCL-cold run followed by an XCL-warm run
# against the same ephemeral XCL data and DXIL cache. The GPU driver's own cache is not
# controlled and is named as such in the manifest.
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$Seconds = 120,
    [string[]]$CargoFeatures = @(),
    [string]$CargoTargetDir
)
$ErrorActionPreference = 'Stop'

if ($Seconds -lt 1) { throw '-Seconds must be at least 1' }
if (-not (Test-Path -LiteralPath $Game -PathType Container)) { throw "game folder not found: $Game" }
$gamePath = [IO.Path]::GetFullPath($Game)
$root = Split-Path $PSScriptRoot -Parent
$workloadAlgorithm = 'sha256-relative-path-length-mtime-utc-ticks'
$workloadAlgorithmVersion = 1

function Get-CanonicalCargoFeatures([string[]]$Values) {
    $features = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($value in @($Values)) {
        foreach ($feature in @($value -split '[,\s]+')) {
            if ([string]::IsNullOrWhiteSpace($feature)) { continue }
            if ($feature.StartsWith('-')) { throw "invalid Cargo feature: $feature" }
            $null = $features.Add($feature)
        }
    }
    [string[]]$canonical = @($features)
    [Array]::Sort($canonical, [StringComparer]::Ordinal)
    return $canonical
}

$cargoFeaturesCanonical = @(Get-CanonicalCargoFeatures $CargoFeatures)

function Read-InventoryRecords([string]$Path) {
    $separators = [char[]]@([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $prefix = $Path.TrimEnd($separators) + [IO.Path]::DirectorySeparatorChar
    $byPath = @{}
    try {
        foreach ($file in Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction Stop) {
            $full = [IO.Path]::GetFullPath($file.FullName)
            if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "inventory path escaped the game folder"
            }
            $info = New-Object IO.FileInfo $full
            $info.Refresh()
            if (-not $info.Exists) { throw "file disappeared during inventory: $($file.Name)" }
            $relative = $full.Substring($prefix.Length).Replace([IO.Path]::DirectorySeparatorChar, '/')
            if ($byPath.ContainsKey($relative)) { throw "duplicate relative inventory path: $relative" }
            $length = $info.Length.ToString([Globalization.CultureInfo]::InvariantCulture)
            $ticks = $info.LastWriteTimeUtc.Ticks.ToString([Globalization.CultureInfo]::InvariantCulture)
            $byPath[$relative] = '{0}:{1}|{2}|{3}' -f $relative.Length, $relative, $length, $ticks
        }
    } catch {
        throw "cannot create a complete game metadata inventory: $($_.Exception.Message)"
    }
    if ($byPath.Count -eq 0) { throw 'cannot fingerprint an empty game folder' }
    [string[]]$paths = @($byPath.Keys)
    [Array]::Sort($paths, [StringComparer]::Ordinal)
    return @($paths | ForEach-Object { $byPath[$_] })
}

function Get-WorkloadIdentity([string]$Path) {
    $first = @(Read-InventoryRecords $Path)
    $second = @(Read-InventoryRecords $Path)
    if ($first.Count -ne $second.Count) { throw 'game metadata inventory changed while it was read' }
    for ($i = 0; $i -lt $first.Count; $i++) {
        if ($first[$i] -cne $second[$i]) { throw 'game metadata inventory changed while it was read' }
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes([string]::Join([char]10, [string[]]$first))
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $id = -join ($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') })
    } finally {
        $sha.Dispose()
    }
    return [ordered]@{ algorithm = $workloadAlgorithm; version = $workloadAlgorithmVersion; id = $id }
}

$workloadIdentity = Get-WorkloadIdentity $gamePath

function Run-Git([string[]]$Arguments) {
    $out = @(& git.exe -C $root @Arguments)
    if ($LASTEXITCODE -ne 0) { throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
    return $out
}

$commit = ((Run-Git -Arguments @('rev-parse', 'HEAD')) -join '').Trim()
$dirtyLines = @(Run-Git -Arguments @('status', '--porcelain', '--untracked-files=all'))
if ($dirtyLines.Count -ne 0) { throw 'perf-run requires a clean worktree so the executable names an exact commit' }

if (Test-Path -LiteralPath $OutDir) {
    if (@(Get-ChildItem -LiteralPath $OutDir -Force).Count -ne 0) { throw "output folder is not empty: $OutDir" }
} else {
    $null = New-Item -ItemType Directory -Path $OutDir
}
$outPath = [IO.Path]::GetFullPath($OutDir)

$cargoTargetKind = 'repository-default'
$cargoTargetPath = Join-Path $root 'target'
if (-not [string]::IsNullOrWhiteSpace($CargoTargetDir)) {
    $cargoTargetKind = 'explicit'
    $cargoTargetPath = if ([IO.Path]::IsPathRooted($CargoTargetDir)) {
        [IO.Path]::GetFullPath($CargoTargetDir)
    } else {
        [IO.Path]::GetFullPath((Join-Path $root $CargoTargetDir))
    }
}
[string[]]$cargoArguments = @('build', '--locked', '--release', '-p', 'xcl')
if ($cargoFeaturesCanonical.Count -ne 0) {
    $cargoArguments += '--features=' + [string]::Join(',', $cargoFeaturesCanonical)
}
$cargoArguments += '--target-dir', $cargoTargetPath

Push-Location $root
try {
    & cargo @cargoArguments
    if ($LASTEXITCODE -ne 0) { throw "release build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
if (@(Run-Git -Arguments @('status', '--porcelain', '--untracked-files=all')).Count -ne 0) {
    throw 'the worktree changed during the release build'
}

$xcl = Join-Path $cargoTargetPath 'release\xcl.exe'
# Test scripts use a command file after still exercising the clean-tree and release-build steps.
if ($env:XCL_PERF_RUN_TESTING -eq '1' -and $env:XCL_PERF_RUN_TEST_XCL) {
    $xcl = $env:XCL_PERF_RUN_TEST_XCL
}
if (-not (Test-Path -LiteralPath $xcl)) { throw "release build did not produce $xcl" }
$xclSha256 = (Get-FileHash -LiteralPath $xcl -Algorithm SHA256).Hash.ToLowerInvariant()

$cacheId = [guid]::NewGuid().ToString('n')
$cacheRoot = Join-Path ([IO.Path]::GetTempPath()) ("xcl-perf-$cacheId")
$dataRoot = Join-Path $cacheRoot 'data'
$dxilCache = Join-Path $cacheRoot 'dxil'
$null = New-Item -ItemType Directory -Path $dataRoot
$null = New-Item -ItemType Directory -Path $dxilCache

$environmentNames = @(
    'XCL_ACTIVITY', 'XCL_ADAPTER', 'XCL_DEVICE_TYPE', 'XCL_DIRECT_DXIL',
    'XCL_DXC_THREADS', 'XCL_FRAME_INTERVAL_US', 'XCL_HEAP_WINDOWS',
    'XCL_GPU_TIMING', 'XCL_GPU_TIMING_MS', 'XCL_LAZY_PSO', 'XCL_NETWORK', 'XCL_PRESS_AT', 'XCL_PSO_PRELOAD',
    'XCL_REQUIRE_GPU', 'XCL_RUN_SECONDS', 'XCL_STALL_TRACE'
)
$changedNames = @('XCL_ACTIVITY', 'XCL_DATA_ROOT', 'XCL_DXIL_CACHE', 'XCL_GPU_TIMING', 'XCL_GPU_TIMING_MS',
    'XCL_NETWORK', 'XCL_RUN_SECONDS', 'XCL_STALL_TRACE')
$savedEnvironment = @{}
foreach ($name in $changedNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Video-Driver([string]$Adapter) {
    try {
        $controllers = @(Get-CimInstance Win32_VideoController)
        $controller = $controllers | Where-Object {
            $Adapter -and ($Adapter.Contains($_.Name) -or $_.Name.Contains($Adapter))
        } | Select-Object -First 1
        if (-not $controller) { $controller = $controllers | Where-Object { $_.Name -notmatch 'Basic' } | Select-Object -First 1 }
        return $controller.DriverVersion
    } catch {
        return $null
    }
}

function Invoke-PerfRun([string]$Mode, [int]$Order) {
    $runName = '{0:d2}-{1}' -f $Order, $Mode
    $artifactDir = Join-Path (Join-Path $outPath 'runs') $runName
    $null = New-Item -ItemType Directory -Path $artifactDir -Force

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    if ($xcl -match '\.(cmd|bat)$') {
        $psi.FileName = 'cmd.exe'
        $psi.Arguments = '/d /c ""' + $xcl + '" run "' + $gamePath + '""'
    } else {
        $psi.FileName = $xcl
        $psi.Arguments = 'run "' + $gamePath + '"'
    }
    $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [Text.Encoding]::UTF8

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $null = $process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $timedOut = -not $process.WaitForExit(($Seconds + 120) * 1000)
    if ($timedOut -and -not $process.HasExited) { $process.Kill() }
    $process.WaitForExit()
    $watch.Stop()
    $stdout = $stdoutTask.Result
    $null = $stderrTask.Result

    $logLine = ($stdout -split '\r?\n' | Where-Object { $_ -match '^Log:\s+(.+)$' } | Select-Object -Last 1)
    if (-not $logLine) { throw "XCL printed no log path during the $Mode run" }
    $log = ($logLine -replace '^Log:\s+', '').Trim()
    if (-not (Test-Path -LiteralPath $log)) { throw "XCL's $Mode log does not exist: $log" }
    Copy-Item -LiteralPath $log -Destination (Join-Path $artifactDir 'xcl.log')

    $sourceReport = Join-Path (Split-Path $log -Parent) 'report.json'
    $reportArtifact = $null
    $adapter = $null
    if (Test-Path -LiteralPath $sourceReport) {
        $reportArtifact = 'report.json'
        Copy-Item -LiteralPath $sourceReport -Destination (Join-Path $artifactDir $reportArtifact)
        try {
            $adapter = ((Get-Content -LiteralPath $sourceReport -Raw -Encoding utf8 | ConvertFrom-Json).d3d12.adapter -replace '\s*\([^()]*VRAM\)$', '')
        } catch { }
    }

    return [ordered]@{
        order = $Order
        cache_state = "xcl-$Mode"
        duration_requested_seconds = $Seconds
        duration_actual_ms = $watch.ElapsedMilliseconds
        exit_code = if ($timedOut) { $null } else { $process.ExitCode }
        timed_out = $timedOut
        gpu = $adapter
        gpu_driver = Video-Driver $adapter
        artifacts = [ordered]@{ log = 'xcl.log'; report = $reportArtifact }
    }
}

try {
    $env:XCL_DATA_ROOT = $dataRoot
    $env:XCL_DXIL_CACHE = $dxilCache
    $env:XCL_NETWORK = '0'
    $env:XCL_RUN_SECONDS = "$Seconds"
    $env:XCL_ACTIVITY = '1'
    $env:XCL_GPU_TIMING = '1'
    $env:XCL_GPU_TIMING_MS = '50'
    $env:XCL_STALL_TRACE = '1'

    $recordedEnvironment = [ordered]@{}
    foreach ($name in $environmentNames) {
        $recordedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    }

    $runs = @()
    $runs += Invoke-PerfRun 'cold' 1
    $runs += Invoke-PerfRun 'warm' 2

    $cpu = $null
    $windows = $null
    try { $cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name } catch { }
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $windows = "$($os.Caption) $($os.Version) build $($os.BuildNumber)"
    } catch { }
    $rust = (& rustc -Vv) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) { throw 'rustc -Vv failed' }
    $verifiedWorkloadIdentity = Get-WorkloadIdentity $gamePath
    if ($verifiedWorkloadIdentity.id -ne $workloadIdentity.id) {
        throw 'game metadata inventory changed during the performance runs'
    }

    $manifest = [ordered]@{
        schema = 'xcl-perf-run'
        version = 1
        commit = $commit
        dirty = $false
        build = [ordered]@{
            profile = 'release'
            command = 'cargo ' + [string]::Join(' ', $cargoArguments)
            arguments = @($cargoArguments)
            cargo_features = $cargoFeaturesCanonical
            target_dir_kind = $cargoTargetKind
            executable_sha256 = $xclSha256
        }
        workload_id = $workloadIdentity.id
        workload_id_algorithm = $workloadIdentity.algorithm
        workload_id_version = $workloadIdentity.version
        run_order = @('xcl-cold', 'xcl-warm')
        system = [ordered]@{ cpu = $cpu; windows = $windows; rust = $rust }
        environment = $recordedEnvironment
        cache = [ordered]@{
            identity = $cacheId
            xcl_data = 'fresh for cold; same ephemeral directory reused for warm'
            dxil = 'fresh for cold; same ephemeral directory reused for warm'
            driver = 'uncontrolled'
        }
        runs = $runs
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outPath 'manifest.json') -Encoding utf8
    Write-Host "Performance artifacts: $outPath"
} finally {
    foreach ($name in $changedNames) {
        $old = $savedEnvironment[$name]
        if ($null -eq $old) {
            Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue
        } else {
            Set-Item -Path "Env:$name" -Value $old
        }
    }
    if (Test-Path -LiteralPath $cacheRoot) {
        Remove-Item -LiteralPath $cacheRoot -Recurse -Force
    }
}
