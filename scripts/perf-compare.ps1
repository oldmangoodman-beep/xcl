# Compare two version-1 perf-run artifact folders without changing either one. Concise text goes
# to the host; the success stream is one machine-readable JSON document.
param(
    [Parameter(Mandatory = $true)][string]$Baseline,
    [Parameter(Mandatory = $true)][string]$Candidate
)
$ErrorActionPreference = 'Stop'

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "missing artifact: $Path" }
    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
    } catch {
        throw "invalid JSON in $($Path): $($_.Exception.Message)"
    }
}

function Has-Property($Object, [string]$Name) {
    return $null -ne $Object -and $null -ne $Object.PSObject.Properties[$Name]
}

function Require-Same-Property($Before, $After, [string]$Name, [string]$Description) {
    $beforeHas = Has-Property $Before $Name
    $afterHas = Has-Property $After $Name
    if ($beforeHas -ne $afterHas) { throw "$Description differs; runs are incomparable" }
    if ($beforeHas) {
        $beforeValue = $Before.$Name | ConvertTo-Json -Compress -Depth 6
        $afterValue = $After.$Name | ConvertTo-Json -Compress -Depth 6
        if ($beforeValue -ne $afterValue) { throw "$Description differs; runs are incomparable" }
    }
}

function Read-Manifest([string]$Root, [string]$Side) {
    $manifest = Read-Json (Join-Path $Root 'manifest.json')
    if ($manifest.schema -ne 'xcl-perf-run' -or [int]$manifest.version -ne 1) {
        throw "$Side has unsupported manifest schema/version"
    }
    if ($manifest.dirty -ne $false -or $manifest.build.profile -ne 'release') {
        throw "$Side is not a clean release run"
    }
    if (Has-Property $manifest.build 'cargo_features') {
        $features = @($manifest.build.cargo_features)
        [string[]]$canonical = @($features | ForEach-Object { [string]$_ })
        [Array]::Sort($canonical, [StringComparer]::Ordinal)
        $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        $malformedFeature = $false
        foreach ($feature in $features) {
            if ([string]::IsNullOrWhiteSpace([string]$feature) -or -not $seen.Add([string]$feature)) {
                $malformedFeature = $true
            }
        }
        if ($malformedFeature -or ($features -join ',') -cne ($canonical -join ',')) {
            throw "$Side has malformed Cargo features"
        }
    }
    if (Has-Property $manifest.build 'executable_sha256') {
        if ([string]$manifest.build.executable_sha256 -cnotmatch '^[0-9a-f]{64}$') {
            throw "$Side has malformed executable identity"
        }
    }
    if ([string]::IsNullOrWhiteSpace([string]$manifest.commit) -or -not (Has-Property $manifest 'environment')) {
        throw "$Side manifest is missing commit/environment"
    }
    if ([string]::IsNullOrWhiteSpace([string]$manifest.workload_id)) {
        throw "$Side has no workload_id"
    }
    if ([string]::IsNullOrWhiteSpace([string]$manifest.workload_id_algorithm) -or
        -not (Has-Property $manifest 'workload_id_version') -or ([int]$manifest.workload_id_version -lt 1)) {
        throw "$Side has no valid workload identity algorithm/version"
    }
    $labels = @($manifest.run_order)
    if ($labels.Count -ne 2 -or $labels[0] -ne 'xcl-cold' -or $labels[1] -ne 'xcl-warm') {
        throw "$Side run_order must be xcl-cold, xcl-warm"
    }
    $runs = @($manifest.runs)
    if ($runs.Count -ne 2) { throw "$Side must contain exactly two run records" }
    for ($i = 0; $i -lt 2; $i++) {
        if ([int]$runs[$i].order -ne $i + 1 -or $runs[$i].cache_state -ne $labels[$i]) {
            throw "$Side run record $($i + 1) does not match run_order"
        }
        if (-not (Has-Property $runs[$i] 'duration_requested_seconds') -or $null -eq $runs[$i].duration_requested_seconds) {
            throw "$Side run record $($i + 1) has no requested duration"
        }
    }
    return $manifest
}

function Read-Stalls([string]$Root, [int]$Order, [string]$Label) {
    $mode = $Label.Substring(4)
    $report = Join-Path $Root (Join-Path 'runs' (Join-Path ('{0:d2}-{1}' -f $Order, $mode) 'report.json'))
    if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
        return [ordered]@{ status = 'missing_report'; aggregates = @{} }
    }
    $json = Read-Json $report
    if (-not (Has-Property $json 'stalls') -or $null -eq $json.stalls) {
        return [ordered]@{ status = 'missing_stalls'; aggregates = @{} }
    }
    if ([int]$json.stalls.version -ne 1) { throw "unsupported stall summary in $report" }
    $aggregates = @{}
    foreach ($aggregate in @($json.stalls.aggregates)) {
        $kind = [string]$aggregate.kind
        if ([string]::IsNullOrWhiteSpace($kind) -or $aggregates.ContainsKey($kind)) {
            throw "invalid or duplicate stall kind in $report"
        }
        $aggregates[$kind] = $aggregate
    }
    return [ordered]@{ status = 'present'; aggregates = $aggregates }
}

function Read-Nonnegative-Integer($Object, [string]$Name, [string]$Context) {
    if (-not (Has-Property $Object $Name) -or $null -eq $Object.$Name -or
        $Object.$Name -is [bool] -or $Object.$Name -is [string]) {
        throw "malformed $Context counter: $Name"
    }
    try { $value = [decimal]$Object.$Name } catch { throw "malformed $Context counter: $Name" }
    if ($value -lt 0 -or $value -ne [decimal]::Truncate($value) -or
        $value -gt [decimal][uint64]::MaxValue) {
        throw "malformed $Context counter: $Name"
    }
    return $value
}

function Read-Nonnegative-Number($Object, [string]$Name, [string]$Context) {
    if (-not (Has-Property $Object $Name) -or $null -eq $Object.$Name -or
        $Object.$Name -is [bool] -or $Object.$Name -is [string]) {
        throw "malformed $Context value: $Name"
    }
    try { $value = [decimal]$Object.$Name } catch { throw "malformed $Context value: $Name" }
    if ($value -lt 0) { throw "malformed $Context value: $Name" }
    return $value
}

function Read-SubmissionTiming([string]$Root, [int]$Order, [string]$Label) {
    $mode = $Label.Substring(4)
    $report = Join-Path $Root (Join-Path 'runs' (Join-Path ('{0:d2}-{1}' -f $Order, $mode) 'report.json'))
    if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
        return [ordered]@{ status = 'missing_report'; counters = $null; histogram_log2_ns = $null }
    }
    $json = Read-Json $report
    if (-not (Has-Property $json 'd3d12') -or $null -eq $json.d3d12) {
        return [ordered]@{ status = 'missing_d3d12'; counters = $null; histogram_log2_ns = $null }
    }
    $legacyNames = @('submissions', 'slowest_submission_ms', 'slowest_submission_lists')
    $extendedNames = @('submission_total_ns', 'submission_mean_ms', 'submission_p95_upper_ms',
        'submission_p99_upper_ms', 'submission_histogram_log2_ns')
    $present = @($legacyNames | Where-Object { (Has-Property $json.d3d12 $_) -and $null -ne $json.d3d12.$_ })
    $extendedPresent = @($extendedNames | Where-Object { (Has-Property $json.d3d12 $_) -and $null -ne $json.d3d12.$_ })
    if ($present.Count -eq 0) {
        if ($extendedPresent.Count -ne 0) { throw "incomplete submission timing in $report" }
        return [ordered]@{ status = 'missing_metrics'; counters = $null; histogram_log2_ns = $null }
    }
    if ($present.Count -ne $legacyNames.Count) { throw "incomplete submission timing in $report" }
    $counters = [ordered]@{
        submissions = Read-Nonnegative-Integer $json.d3d12 'submissions' "submission timing in $report"
        slowest_submission_ms = Read-Nonnegative-Number $json.d3d12 'slowest_submission_ms' "submission timing in $report"
        slowest_submission_lists = Read-Nonnegative-Integer $json.d3d12 'slowest_submission_lists' "submission timing in $report"
    }
    if ($extendedPresent.Count -eq 0) {
        return [ordered]@{ status = 'present_legacy'; counters = $counters; histogram_log2_ns = $null }
    }
    if ($extendedPresent.Count -ne $extendedNames.Count) { throw "incomplete extended submission timing in $report" }
    $counters['submission_total_ns'] = Read-Nonnegative-Integer $json.d3d12 'submission_total_ns' "submission timing in $report"
    $counters['submission_mean_ms'] = Read-Nonnegative-Number $json.d3d12 'submission_mean_ms' "submission timing in $report"
    $counters['submission_p95_upper_ms'] = Read-Nonnegative-Number $json.d3d12 'submission_p95_upper_ms' "submission timing in $report"
    $counters['submission_p99_upper_ms'] = Read-Nonnegative-Number $json.d3d12 'submission_p99_upper_ms' "submission timing in $report"
    $histogram = @($json.d3d12.submission_histogram_log2_ns)
    if ($histogram.Count -ne 64) { throw "malformed submission timing histogram in $report" }
    $histogram = @($histogram | ForEach-Object {
        $value = [pscustomobject]@{ value = $_ }
        Read-Nonnegative-Integer $value 'value' "submission timing histogram in $report"
    })
    $histogramCount = [decimal]0
    foreach ($count in $histogram) { $histogramCount += $count }
    if ($histogramCount -ne $counters.submissions) { throw "submission timing histogram count differs in $report" }
    return [ordered]@{ status = 'present'; counters = $counters; histogram_log2_ns = $histogram }
}

function Read-Preparation([string]$Root, [int]$Order, [string]$Label) {
    $mode = $Label.Substring(4)
    $report = Join-Path $Root (Join-Path 'runs' (Join-Path ('{0:d2}-{1}' -f $Order, $mode) 'report.json'))
    if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
        return [ordered]@{ status = 'missing_report'; counters = $null; ranked_status = 'missing_report'; ranked = $null; allocator_status = 'missing_report'; allocator = $null }
    }
    $json = Read-Json $report
    if (-not (Has-Property $json 'preparation') -or $null -eq $json.preparation) {
        return [ordered]@{ status = 'missing_preparation'; counters = $null; ranked_status = 'missing_preparation'; ranked = $null; allocator_status = 'missing_preparation'; allocator = $null }
    }
    $summary = $json.preparation
    $version = Read-Nonnegative-Integer $summary 'version' 'preparation version'
    if ($version -eq 0) { throw "malformed preparation version in $report" }
    if ($version -ne 1 -and $version -ne 2 -and $version -ne 3) {
        return [ordered]@{ status = 'unsupported_version'; counters = $null; ranked_status = 'unsupported_version'; ranked = $null; allocator_status = 'unsupported_version'; allocator = $null }
    }
    if (-not (Has-Property $summary 'compute_build_coalescing') -or
        $null -eq $summary.compute_build_coalescing) {
        throw "malformed preparation counters in $report"
    }
    $raw = $summary.compute_build_coalescing
    $counters = [ordered]@{}
    foreach ($name in @('keyed_submissions', 'coalescing_hits', 'unique_keyed_builds',
            'current_in_flight', 'peak_in_flight')) {
        $counters[$name] = Read-Nonnegative-Integer $raw $name "preparation in $report"
    }
    if ($version -eq 1) {
        return [ordered]@{ status = 'present'; counters = $counters; ranked_status = 'not_in_version'; ranked = $null; allocator_status = 'not_in_version'; allocator = $null }
    }
    if (-not (Has-Property $summary 'ranked_preloads') -or $null -eq $summary.ranked_preloads) {
        throw "malformed ranked preload counters in $report"
    }
    $ranked = [ordered]@{}
    foreach ($name in @('submissions', 'started', 'promoted')) {
        $ranked[$name] = Read-Nonnegative-Integer $summary.ranked_preloads $name "ranked preloads in $report"
    }
    if ($version -eq 2) {
        return [ordered]@{ status = 'present'; counters = $counters; ranked_status = 'present'; ranked = $ranked; allocator_status = 'not_in_version'; allocator = $null }
    }
    if (-not (Has-Property $summary 'allocator_submits') -or $null -eq $summary.allocator_submits) {
        throw "malformed allocator submit counters in $report"
    }
    $allocator = [ordered]@{}
    foreach ($name in @('outer_execute_calls', 'chunks', 'multi_chunk_calls', 'forced_wait_boundaries',
            'lists_marked', 'backing_mark_writes', 'signal_attempts', 'signal_failures',
            'state_lock_wait_ns', 'state_lock_hold_ns')) {
        $allocator[$name] = Read-Nonnegative-Integer $summary.allocator_submits $name "allocator submits in $report"
    }
    return [ordered]@{ status = 'present'; counters = $counters; ranked_status = 'present'; ranked = $ranked; allocator_status = 'present'; allocator = $allocator }
}

function Read-SparseUploadGate([string]$Root, [int]$Order, [string]$Label) {
    $mode = $Label.Substring(4)
    $log = Join-Path $Root (Join-Path 'runs' (Join-Path ('{0:d2}-{1}' -f $Order, $mode) 'xcl.log'))
    if (-not (Test-Path -LiteralPath $log -PathType Leaf)) {
        return [ordered]@{ status = 'missing_log'; counters = $null }
    }
    $text = Get-Content -LiteralPath $log -Raw -Encoding utf8
    $marker = 'upload gate submit wait/hold'
    $pattern = 'upload gate submit wait/hold (?<submit_wait>\d+)/(?<submit_hold>\d+) ms, readback wait/hold/released wait\+reacquire (?<readback_wait>\d+)/(?<readback_hold>\d+)/(?<readback_released>\d+) ms'
    $matches = [regex]::Matches($text, $pattern)
    $markers = [regex]::Matches($text, [regex]::Escape($marker))
    if ($matches.Count -ne $markers.Count) {
        throw "malformed sparse upload-gate metrics in $log"
    }
    if ($matches.Count -eq 0) {
        return [ordered]@{ status = 'missing_metrics'; counters = $null }
    }
    $counters = [ordered]@{
        submit_wait_ms = [decimal]0
        submit_hold_ms = [decimal]0
        readback_wait_ms = [decimal]0
        readback_hold_ms = [decimal]0
        readback_released_wait_reacquire_ms = [decimal]0
    }
    foreach ($match in $matches) {
        $counters.submit_wait_ms += [decimal]$match.Groups['submit_wait'].Value
        $counters.submit_hold_ms += [decimal]$match.Groups['submit_hold'].Value
        $counters.readback_wait_ms += [decimal]$match.Groups['readback_wait'].Value
        $counters.readback_hold_ms += [decimal]$match.Groups['readback_hold'].Value
        $counters.readback_released_wait_reacquire_ms += [decimal]$match.Groups['readback_released'].Value
    }
    return [ordered]@{ status = 'present'; counters = $counters }
}

function Number-Or-Null($Object, [string]$Name) {
    if ($null -eq $Object -or -not (Has-Property $Object $Name) -or $null -eq $Object.$Name) { return $null }
    return [long]$Object.$Name
}

function Delta-Or-Null($Before, $After) {
    if ($null -eq $Before -or $null -eq $After) { return $null }
    return [long]$After - [long]$Before
}

function Counter-Delta-Or-Null($Before, $After) {
    if ($null -eq $Before -or $null -eq $After) { return $null }
    return [decimal]$After - [decimal]$Before
}

function Compare-Counters($Before, $After, [string[]]$Names) {
    $delta = [ordered]@{}
    foreach ($name in $Names) {
        $delta[$name] = Counter-Delta-Or-Null $Before.$name $After.$name
    }
    return [ordered]@{ baseline = $Before; candidate = $After; delta = $delta }
}

function Bool-Or-Null($Object, [string]$Name) {
    if ($null -eq $Object -or -not (Has-Property $Object $Name) -or $null -eq $Object.$Name) { return $null }
    return [bool]$Object.$Name
}

function Aggregate-Or-Null($Aggregate) {
    if ($null -eq $Aggregate) { return $null }
    return [ordered]@{
        count = Number-Or-Null $Aggregate 'count'
        total_ns = Number-Or-Null $Aggregate 'total_ns'
        max_ns = Number-Or-Null $Aggregate 'max_ns'
        histogram_log2_ns = @($Aggregate.histogram_log2_ns | ForEach-Object { [long]$_ })
    }
}

function Compare-Aggregates($BeforeSet, $AfterSet) {
    $kinds = @($BeforeSet.Keys) + @($AfterSet.Keys) | Sort-Object -Unique
    $out = @()
    foreach ($kind in $kinds) {
        $beforeRaw = if ($BeforeSet.ContainsKey($kind)) { $BeforeSet[$kind] } else { $null }
        $afterRaw = if ($AfterSet.ContainsKey($kind)) { $AfterSet[$kind] } else { $null }
        $before = Aggregate-Or-Null $beforeRaw
        $after = Aggregate-Or-Null $afterRaw
        $histogramDelta = $null
        if ($null -ne $before -and $null -ne $after -and $before.histogram_log2_ns.Count -eq $after.histogram_log2_ns.Count) {
            $histogramDelta = for ($i = 0; $i -lt $before.histogram_log2_ns.Count; $i++) {
                $after.histogram_log2_ns[$i] - $before.histogram_log2_ns[$i]
            }
        }
        $out += [ordered]@{
            kind = $kind
            baseline = $before
            candidate = $after
            delta = [ordered]@{
                count = Delta-Or-Null $before.count $after.count
                total_ns = Delta-Or-Null $before.total_ns $after.total_ns
                max_ns = Delta-Or-Null $before.max_ns $after.max_ns
                histogram_log2_ns = $histogramDelta
            }
        }
    }
    return $out
}

function Display-Value($Value, [string]$Suffix) {
    if ($null -eq $Value) { return 'missing' }
    return "$Value$Suffix"
}

$baselineRoot = [IO.Path]::GetFullPath($Baseline)
$candidateRoot = [IO.Path]::GetFullPath($Candidate)
$beforeManifest = Read-Manifest $baselineRoot 'baseline'
$afterManifest = Read-Manifest $candidateRoot 'candidate'

if ($beforeManifest.workload_id -cne $afterManifest.workload_id) { throw 'workload_id differs; runs are incomparable' }
if ($beforeManifest.workload_id_algorithm -cne $afterManifest.workload_id_algorithm) {
    throw 'workload identity algorithm differs; runs are incomparable'
}
if ([int]$beforeManifest.workload_id_version -ne [int]$afterManifest.workload_id_version) {
    throw 'workload identity version differs; runs are incomparable'
}
if ((@($beforeManifest.run_order) -join ',') -ne (@($afterManifest.run_order) -join ',')) { throw 'run order differs; runs are incomparable' }
Require-Same-Property $beforeManifest $afterManifest 'system' 'system information'
if (($beforeManifest.environment | ConvertTo-Json -Compress -Depth 6) -ne ($afterManifest.environment | ConvertTo-Json -Compress -Depth 6)) {
    throw 'runtime environment differs; runs are incomparable'
}

$comparisons = @()
for ($i = 0; $i -lt 2; $i++) {
    $beforeRun = @($beforeManifest.runs)[$i]
    $afterRun = @($afterManifest.runs)[$i]
    Require-Same-Property $beforeRun $afterRun 'gpu' "$($beforeRun.cache_state) GPU"
    Require-Same-Property $beforeRun $afterRun 'gpu_driver' "$($beforeRun.cache_state) GPU driver"
    if ([long]$beforeRun.duration_requested_seconds -ne [long]$afterRun.duration_requested_seconds) {
        throw "$($beforeRun.cache_state) requested duration differs; runs are incomparable"
    }
    $beforeDuration = Number-Or-Null $beforeRun 'duration_actual_ms'
    $afterDuration = Number-Or-Null $afterRun 'duration_actual_ms'
    $beforeStalls = Read-Stalls $baselineRoot ($i + 1) $beforeRun.cache_state
    $afterStalls = Read-Stalls $candidateRoot ($i + 1) $afterRun.cache_state
    $stalls = @(Compare-Aggregates $beforeStalls.aggregates $afterStalls.aggregates)
    $beforePreparation = Read-Preparation $baselineRoot ($i + 1) $beforeRun.cache_state
    $afterPreparation = Read-Preparation $candidateRoot ($i + 1) $afterRun.cache_state
    $coalescingNames = @('keyed_submissions', 'coalescing_hits', 'unique_keyed_builds', 'current_in_flight', 'peak_in_flight')
    $coalescing = Compare-Counters $beforePreparation.counters $afterPreparation.counters $coalescingNames
    $rankedPreloads = Compare-Counters $beforePreparation.ranked $afterPreparation.ranked @('submissions', 'started', 'promoted')
    $allocatorNames = @('outer_execute_calls', 'chunks', 'multi_chunk_calls', 'forced_wait_boundaries',
        'lists_marked', 'backing_mark_writes', 'signal_attempts', 'signal_failures',
        'state_lock_wait_ns', 'state_lock_hold_ns')
    $allocatorSubmits = Compare-Counters $beforePreparation.allocator $afterPreparation.allocator $allocatorNames
    $beforeSparseGate = Read-SparseUploadGate $baselineRoot ($i + 1) $beforeRun.cache_state
    $afterSparseGate = Read-SparseUploadGate $candidateRoot ($i + 1) $afterRun.cache_state
    $sparseGateNames = @('submit_wait_ms', 'submit_hold_ms', 'readback_wait_ms', 'readback_hold_ms',
        'readback_released_wait_reacquire_ms')
    $sparseUploadGate = Compare-Counters $beforeSparseGate.counters $afterSparseGate.counters $sparseGateNames
    $beforeSubmissionTiming = Read-SubmissionTiming $baselineRoot ($i + 1) $beforeRun.cache_state
    $afterSubmissionTiming = Read-SubmissionTiming $candidateRoot ($i + 1) $afterRun.cache_state
    $submissionTimingNames = @('submissions', 'slowest_submission_ms', 'slowest_submission_lists',
        'submission_total_ns', 'submission_mean_ms', 'submission_p95_upper_ms', 'submission_p99_upper_ms')
    $submissionTiming = Compare-Counters $beforeSubmissionTiming.counters $afterSubmissionTiming.counters $submissionTimingNames

    $comparison = [ordered]@{
        label = $beforeRun.cache_state
        duration_ms = [ordered]@{
            baseline = $beforeDuration
            candidate = $afterDuration
            delta = Delta-Or-Null $beforeDuration $afterDuration
        }
        exit_code = [ordered]@{
            baseline = Number-Or-Null $beforeRun 'exit_code'
            candidate = Number-Or-Null $afterRun 'exit_code'
        }
        timed_out = [ordered]@{
            baseline = Bool-Or-Null $beforeRun 'timed_out'
            candidate = Bool-Or-Null $afterRun 'timed_out'
        }
        stalls_status = [ordered]@{ baseline = $beforeStalls.status; candidate = $afterStalls.status }
        stalls = $stalls
        preparation_status = [ordered]@{
            baseline = $beforePreparation.status
            candidate = $afterPreparation.status
        }
        compute_build_coalescing = $coalescing
        ranked_preloads_status = [ordered]@{
            baseline = $beforePreparation.ranked_status
            candidate = $afterPreparation.ranked_status
        }
        ranked_preloads = $rankedPreloads
        allocator_submits_status = [ordered]@{
            baseline = $beforePreparation.allocator_status
            candidate = $afterPreparation.allocator_status
        }
        allocator_submits = $allocatorSubmits
        sparse_upload_gate_status = [ordered]@{
            baseline = $beforeSparseGate.status
            candidate = $afterSparseGate.status
        }
        sparse_upload_gate = $sparseUploadGate
        submission_timing_status = [ordered]@{
            baseline = $beforeSubmissionTiming.status
            candidate = $afterSubmissionTiming.status
        }
        submission_timing = $submissionTiming
        submission_timing_histogram_log2_ns = [ordered]@{
            baseline = $beforeSubmissionTiming.histogram_log2_ns
            candidate = $afterSubmissionTiming.histogram_log2_ns
        }
    }
    $comparisons += $comparison

    Write-Host "$($comparison.label): duration $(Display-Value $beforeDuration ' ms') -> $(Display-Value $afterDuration ' ms'); delta $(Display-Value $comparison.duration_ms.delta ' ms')"
    Write-Host "  exit: $(Display-Value $comparison.exit_code.baseline '') -> $(Display-Value $comparison.exit_code.candidate ''); timeout: $(Display-Value $comparison.timed_out.baseline '') -> $(Display-Value $comparison.timed_out.candidate '')"
    if ($stalls.Count -eq 0) {
        Write-Host "  stalls: missing/empty (baseline $($beforeStalls.status), candidate $($afterStalls.status))"
    } else {
        foreach ($stall in $stalls) {
            Write-Host "  stall $($stall.kind): total $(Display-Value $stall.baseline.total_ns ' ns') -> $(Display-Value $stall.candidate.total_ns ' ns'); count $(Display-Value $stall.baseline.count '') -> $(Display-Value $stall.candidate.count ''); max $(Display-Value $stall.baseline.max_ns ' ns') -> $(Display-Value $stall.candidate.max_ns ' ns')"
        }
    }
    Write-Host "  preparation: $($beforePreparation.status) -> $($afterPreparation.status)"
    if ($beforePreparation.status -eq 'present' -or $afterPreparation.status -eq 'present') {
        foreach ($name in $coalescing.delta.Keys) {
            Write-Host "    $($name): $(Display-Value $coalescing.baseline.$name '') -> $(Display-Value $coalescing.candidate.$name ''); delta $(Display-Value $coalescing.delta.$name '')"
        }
    }
    Write-Host "  ranked preloads: $($beforePreparation.ranked_status) -> $($afterPreparation.ranked_status)"
    if ($beforePreparation.ranked_status -eq 'present' -or $afterPreparation.ranked_status -eq 'present') {
        foreach ($name in $rankedPreloads.delta.Keys) {
            Write-Host "    $($name): $(Display-Value $rankedPreloads.baseline.$name '') -> $(Display-Value $rankedPreloads.candidate.$name ''); delta $(Display-Value $rankedPreloads.delta.$name '')"
        }
    }
    Write-Host "  allocator submits: $($beforePreparation.allocator_status) -> $($afterPreparation.allocator_status)"
    if ($beforePreparation.allocator_status -eq 'present' -or $afterPreparation.allocator_status -eq 'present') {
        foreach ($name in $allocatorSubmits.delta.Keys) {
            Write-Host "    $($name): $(Display-Value $allocatorSubmits.baseline.$name '') -> $(Display-Value $allocatorSubmits.candidate.$name ''); delta $(Display-Value $allocatorSubmits.delta.$name '')"
        }
    }
    Write-Host "  sparse upload gate: $($beforeSparseGate.status) -> $($afterSparseGate.status)"
    if ($beforeSparseGate.status -eq 'present' -or $afterSparseGate.status -eq 'present') {
        foreach ($name in $sparseUploadGate.delta.Keys) {
            Write-Host "    $($name): $(Display-Value $sparseUploadGate.baseline.$name ' ms') -> $(Display-Value $sparseUploadGate.candidate.$name ' ms'); delta $(Display-Value $sparseUploadGate.delta.$name ' ms')"
        }
    }
    Write-Host "  submission timing: $($beforeSubmissionTiming.status) -> $($afterSubmissionTiming.status)"
    if ($beforeSubmissionTiming.status -like 'present*' -or $afterSubmissionTiming.status -like 'present*') {
        foreach ($name in $submissionTiming.delta.Keys) {
            $suffix = if ($name -like '*_ms') { ' ms' } elseif ($name -eq 'submission_total_ns') { ' ns' } else { '' }
            Write-Host "    $($name): $(Display-Value $submissionTiming.baseline.$name $suffix) -> $(Display-Value $submissionTiming.candidate.$name $suffix); delta $(Display-Value $submissionTiming.delta.$name $suffix)"
        }
    }
}

[ordered]@{
    schema = 'xcl-perf-comparison'
    version = 1
    workload_id = $beforeManifest.workload_id
    workload_id_algorithm = $beforeManifest.workload_id_algorithm
    workload_id_version = [int]$beforeManifest.workload_id_version
    run_order = @($beforeManifest.run_order)
    baseline = [ordered]@{ commit = $beforeManifest.commit; path = $baselineRoot; build = $beforeManifest.build }
    candidate = [ordered]@{ commit = $afterManifest.commit; path = $candidateRoot; build = $afterManifest.build }
    runs = $comparisons
} | ConvertTo-Json -Depth 12
