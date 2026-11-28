# Run on Windows with powershell.exe or pwsh.exe. Uses a clean temporary repository and mock
# cargo/XCL commands; no game executable is run.
$ErrorActionPreference = 'Stop'
$runnerSource = Join-Path $PSScriptRoot 'perf-run.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-perf-run-test-' + [guid]::NewGuid())
$repo = Join-Path $fixture 'repo'
$scripts = Join-Path $repo 'scripts'
$game = Join-Path $fixture 'Mock Game'
$out = Join-Path $fixture 'artifacts'
$otherGame = Join-Path (Join-Path $fixture 'other-build') 'Mock Game'
$otherOut = Join-Path $fixture 'other-artifacts'
$featureOut = Join-Path $fixture 'feature-artifacts'
$featureTarget = Join-Path $fixture 'feature target'
$bin = Join-Path $fixture 'bin'
$userData = Join-Path $fixture 'user-data'
$userDxil = Join-Path $fixture 'user-dxil'
$cargoMarker = Join-Path $fixture 'cargo.txt'
$mockXcl = Join-Path $bin 'mock-xcl.cmd'
$oldPath = $env:PATH
$oldTest = $env:XCL_PERF_RUN_TESTING
$oldTestXcl = $env:XCL_PERF_RUN_TEST_XCL
$oldData = $env:XCL_DATA_ROOT
$oldDxil = $env:XCL_DXIL_CACHE
$oldPress = $env:XCL_PRESS_AT
$oldGpuTiming = $env:XCL_GPU_TIMING
$oldGpuTimingMs = $env:XCL_GPU_TIMING_MS

try {
    $null = New-Item -ItemType Directory -Path $scripts, $game, $bin, $userData, $userDxil
    Copy-Item -LiteralPath $runnerSource -Destination (Join-Path $scripts 'perf-run.ps1')
    'target/' | Set-Content -LiteralPath (Join-Path $repo '.gitignore') -Encoding Ascii
    'not copied' | Set-Content -LiteralPath (Join-Path $game 'game-data.bin') -Encoding Ascii
    'keep' | Set-Content -LiteralPath (Join-Path $userData 'sentinel.txt') -Encoding Ascii
    'keep' | Set-Content -LiteralPath (Join-Path $userDxil 'sentinel.txt') -Encoding Ascii

    @'
@echo off
echo %* > "%MOCK_CARGO_MARKER%"
exit /b 0
'@ | Set-Content -LiteralPath (Join-Path $bin 'cargo.cmd') -Encoding Ascii
    @'
@echo off
echo rustc 1.test
echo host: x86_64-pc-windows-msvc
exit /b 0
'@ | Set-Content -LiteralPath (Join-Path $bin 'rustc.cmd') -Encoding Ascii
    @'
@echo off
set STATE=warm
if not exist "%XCL_DXIL_CACHE%\seen" set STATE=cold
if exist "%XCL_DXIL_CACHE%\seen" goto marker_done
echo seen>"%XCL_DXIL_CACHE%\seen"
:marker_done
if not exist "%XCL_DATA_ROOT%\MockTitle\logs" mkdir "%XCL_DATA_ROOT%\MockTitle\logs"
echo cache=%STATE%>"%XCL_DATA_ROOT%\MockTitle\logs\xcl.log"
echo gpu_timing=%XCL_GPU_TIMING% threshold=%XCL_GPU_TIMING_MS%>>"%XCL_DATA_ROOT%\MockTitle\logs\xcl.log"
echo {"d3d12":{"adapter":"Mock GPU","submissions":2,"slowest_submission_ms":1.5,"slowest_submission_lists":3}}>"%XCL_DATA_ROOT%\MockTitle\logs\report.json"
echo Log: %XCL_DATA_ROOT%\MockTitle\logs\xcl.log
exit /b 0
'@ | Set-Content -LiteralPath $mockXcl -Encoding Ascii

    & git.exe -C $repo init | Out-Null
    & git.exe -C $repo add .
    & git.exe -C $repo -c user.name='XCL test' -c user.email='xcl-test@example.invalid' -c commit.gpgsign=false commit -m fixture | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'could not create clean fixture repository' }

    $env:PATH = $bin + [IO.Path]::PathSeparator + $oldPath
    $env:MOCK_CARGO_MARKER = $cargoMarker
    $env:XCL_PERF_RUN_TESTING = '1'
    $env:XCL_PERF_RUN_TEST_XCL = $mockXcl
    $env:XCL_DATA_ROOT = $userData
    $env:XCL_DXIL_CACHE = $userDxil
    $env:XCL_PRESS_AT = '20:A,24:A'
    $env:XCL_GPU_TIMING = 'caller-value'
    $env:XCL_GPU_TIMING_MS = '999'

    & (Join-Path $scripts 'perf-run.ps1') -Game $game -OutDir $out -Seconds 2
    if ($LASTEXITCODE -ne 0) { throw "perf-run.ps1 exited with $LASTEXITCODE" }

    $manifest = Get-Content -LiteralPath (Join-Path $out 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $manifestRaw = Get-Content -LiteralPath (Join-Path $out 'manifest.json') -Raw -Encoding utf8
    if ($manifest.schema -ne 'xcl-perf-run' -or $manifest.version -ne 1) { throw 'wrong manifest schema/version' }
    if ($manifest.dirty -ne $false -or $manifest.build.profile -ne 'release') { throw 'build identity was not recorded' }
    if (($manifest.run_order -join ',') -ne 'xcl-cold,xcl-warm') { throw 'run order was not cold then warm' }
    if ($manifest.cache.driver -ne 'uncontrolled') { throw 'driver cache was not labelled uncontrolled' }
    if ($manifest.environment.XCL_PRESS_AT -ne '20:A,24:A') { throw 'input schedule was not recorded' }
    if ($manifest.environment.XCL_GPU_TIMING -ne '1' -or $manifest.environment.XCL_GPU_TIMING_MS -ne '50') {
        throw 'submission timing settings were not forced and recorded'
    }
    if ($manifest.workload_id_algorithm -ne 'sha256-relative-path-length-mtime-utc-ticks' -or
        $manifest.workload_id_version -ne 1) { throw 'workload identity algorithm/version was not recorded' }
    if ($manifest.workload_id -notmatch '^[0-9a-f]{64}$') { throw 'workload identity is not a full SHA-256 value' }
    if ($manifestRaw.Contains($game) -or $manifestRaw.Contains('game-data.bin')) {
        throw 'workload inventory details leaked into the manifest'
    }
    if ($manifest.runs.Count -ne 2 -or $manifest.runs[0].cache_state -ne 'xcl-cold' -or $manifest.runs[1].cache_state -ne 'xcl-warm') { throw 'run records are incomplete' }
    if ((Get-Content -LiteralPath (Join-Path $out 'runs\01-cold\xcl.log') -Raw) -notmatch 'cache=cold') { throw 'cold run did not start with a fresh DXIL cache' }
    if ((Get-Content -LiteralPath (Join-Path $out 'runs\02-warm\xcl.log') -Raw) -notmatch 'cache=warm') { throw 'warm run did not reuse the cold cache' }
    if ((Get-Content -LiteralPath (Join-Path $out 'runs\01-cold\xcl.log') -Raw) -notmatch 'gpu_timing=1 threshold=50') {
        throw 'submission timing settings were not passed to XCL'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $out 'runs\01-cold\report.json')) -or -not (Test-Path -LiteralPath (Join-Path $out 'runs\02-warm\report.json'))) { throw 'reports were not preserved' }
    $timing = (Get-Content -LiteralPath (Join-Path $out 'runs\01-cold\report.json') -Raw -Encoding utf8 | ConvertFrom-Json).d3d12
    if ($timing.submissions -ne 2 -or $timing.slowest_submission_ms -ne 1.5 -or $timing.slowest_submission_lists -ne 3) {
        throw 'submission timing fields were not preserved'
    }
    $artifactFiles = @(Get-ChildItem -LiteralPath $out -Recurse -File)
    $unexpectedArtifacts = @($artifactFiles | Where-Object { $_.Name -notin @('manifest.json', 'xcl.log', 'report.json') })
    if ($artifactFiles.Count -ne 5 -or $unexpectedArtifacts.Count -ne 0) { throw 'the artifact folder contains something other than logs, reports, and the manifest' }
    if ($env:XCL_DATA_ROOT -ne $userData -or $env:XCL_DXIL_CACHE -ne $userDxil -or
        $env:XCL_GPU_TIMING -ne 'caller-value' -or $env:XCL_GPU_TIMING_MS -ne '999') {
        throw 'the caller environment was not restored'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $userData 'sentinel.txt')) -or -not (Test-Path -LiteralPath (Join-Path $userDxil 'sentinel.txt'))) { throw 'a user cache was touched' }
    if (Test-Path -LiteralPath (Join-Path ([IO.Path]::GetTempPath()) ("xcl-perf-$($manifest.cache.identity)"))) { throw 'the ephemeral cache was not removed' }
    if ((Get-Content -LiteralPath $cargoMarker -Raw) -notmatch 'build --locked --release -p xcl') { throw 'release build command was not run' }

    & (Join-Path $scripts 'perf-run.ps1') -Game $game -OutDir $featureOut -Seconds 2 `
        -CargoFeatures @('zeta,alpha', 'alpha') -CargoTargetDir $featureTarget
    if ($LASTEXITCODE -ne 0) { throw "feature perf-run.ps1 exited with $LASTEXITCODE" }
    $featureManifest = Get-Content -LiteralPath (Join-Path $featureOut 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ((@($featureManifest.build.cargo_features) -join ',') -ne 'alpha,zeta' -or
        $featureManifest.build.target_dir_kind -ne 'explicit' -or
        $featureManifest.build.executable_sha256 -notmatch '^[0-9a-f]{64}$') {
        throw 'the canonical Cargo build variant was not recorded'
    }
    $cargoInvocation = Get-Content -LiteralPath $cargoMarker -Raw
    if ($cargoInvocation -notmatch '--features=alpha,zeta' -or
        $cargoInvocation -notmatch '--target-dir' -or
        $cargoInvocation -notmatch ([regex]::Escape($featureTarget))) {
        throw 'Cargo features/target directory were not passed as canonical arguments'
    }

    $null = New-Item -ItemType Directory -Path $otherGame -Force
    'a different title build with the same folder leaf' | Set-Content -LiteralPath (Join-Path $otherGame 'game-data.bin') -Encoding Ascii
    & (Join-Path $scripts 'perf-run.ps1') -Game $otherGame -OutDir $otherOut -Seconds 2
    if ($LASTEXITCODE -ne 0) { throw "second perf-run.ps1 exited with $LASTEXITCODE" }
    $otherManifest = Get-Content -LiteralPath (Join-Path $otherOut 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
    if ($otherManifest.workload_id -eq $manifest.workload_id) {
        throw 'different builds with the same folder leaf received the same workload identity'
    }

    Add-Content -LiteralPath (Join-Path $repo '.gitignore') -Value '# dirty'
    $refused = $false
    try {
        & (Join-Path $scripts 'perf-run.ps1') -Game $game -OutDir (Join-Path $fixture 'dirty-out') -Seconds 2
    } catch {
        $refused = $_.Exception.Message -match 'clean worktree'
    }
    if (-not $refused) { throw 'a dirty worktree was not refused' }

    Write-Host 'perf-run tests: all assertions passed'
} finally {
    $env:PATH = $oldPath
    $env:XCL_PERF_RUN_TESTING = $oldTest
    $env:XCL_PERF_RUN_TEST_XCL = $oldTestXcl
    $env:XCL_DATA_ROOT = $oldData
    $env:XCL_DXIL_CACHE = $oldDxil
    $env:XCL_PRESS_AT = $oldPress
    $env:XCL_GPU_TIMING = $oldGpuTiming
    $env:XCL_GPU_TIMING_MS = $oldGpuTimingMs
    Remove-Item Env:MOCK_CARGO_MARKER -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
}
