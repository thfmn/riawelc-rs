<script lang="ts">
	import ImageUpload from '$lib/components/ImageUpload.svelte';
	import HeatmapOverlay from '$lib/components/HeatmapOverlay.svelte';
	import SegmentationMask from '$lib/components/SegmentationMask.svelte';
	import ConfidenceChart from '$lib/components/ConfidenceChart.svelte';
	import ImageStatistics from '$lib/components/ImageStatistics.svelte';
	import ModelInfo from '$lib/components/ModelInfo.svelte';
	import {
		predict,
		segmentUnet,
		isApiError,
		type PredictResponse,
		type UNetSegmentResponse
	} from '$lib/api';

	let loading = $state(false);
	let error: string | null = $state(null);

	let originalSrc: string | null = $state(null);
	let predictResult: PredictResponse | null = $state(null);
	let segResult: UNetSegmentResponse | null = $state(null);
	let defectAreaPct: number | null = $state(null);

	let hasResults = $derived(predictResult !== null || segResult !== null);

	async function handleFileSelected(file: File) {
		error = null;
		predictResult = null;
		segResult = null;
		defectAreaPct = null;
		loading = true;

		if (originalSrc) URL.revokeObjectURL(originalSrc);
		originalSrc = URL.createObjectURL(file);

		try {
			const [pResult, sResult] = await Promise.allSettled([
				predict(file),
				segmentUnet(file)
			]);

			if (pResult.status === 'fulfilled') {
				predictResult = pResult.value;
			} else {
				error = isApiError(pResult.reason) ? pResult.reason.message : 'Classification failed';
			}

			if (sResult.status === 'fulfilled') {
				segResult = sResult.value;
			} else if (!error) {
				error = isApiError(sResult.reason) ? sResult.reason.message : 'Segmentation failed';
			}
		} catch (err: unknown) {
			error = isApiError(err) ? err.message : 'An unexpected error occurred';
		} finally {
			loading = false;
		}
	}

	function handleReset() {
		if (originalSrc) URL.revokeObjectURL(originalSrc);
		originalSrc = null;
		predictResult = null;
		segResult = null;
		defectAreaPct = null;
		error = null;
		loading = false;
	}
</script>

<div class="page-layout">
	<aside class="sidebar">
		<div class="sidebar-content">
			<div class="sidebar-section">
				<h2 class="section-label">Upload Image</h2>
				<ImageUpload onFileSelected={handleFileSelected} disabled={loading} />

				{#if hasResults || loading}
					<button class="btn-reset" onclick={handleReset} disabled={loading} aria-label="Reset analysis">
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
							<polyline points="1 4 1 10 7 10" />
							<path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
						</svg>
						Reset
					</button>
				{/if}
			</div>

			<div class="sidebar-section">
				<ModelInfo />
			</div>

			<div class="sidebar-footer">
				<span class="footer-text">RIAWELC Inspector</span>
			</div>
		</div>
	</aside>

	<div class="main-area">
		{#if loading}
			<div class="state-screen">
				<div class="spinner"></div>
				<h2 class="state-title">Analyzing Image</h2>
				<p class="state-desc">Running classification, Grad-CAM, and segmentation...</p>
			</div>
		{:else if error && !hasResults}
			<div class="state-screen">
				<div class="error-icon">
					<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
						<circle cx="12" cy="12" r="10" />
						<line x1="15" y1="9" x2="9" y2="15" />
						<line x1="9" y1="9" x2="15" y2="15" />
					</svg>
				</div>
				<h2 class="state-title">Analysis Failed</h2>
				<p class="state-desc">{error}</p>
				<button class="btn-retry" onclick={handleReset}>
					Try Again
				</button>
			</div>
		{:else if hasResults && originalSrc}
			<div class="results-grid animate-fade-in">
				{#if error}
					<div class="partial-error" role="alert">
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
							<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
							<line x1="12" y1="9" x2="12" y2="13" />
							<line x1="12" y1="17" x2="12.01" y2="17" />
						</svg>
						<span>Partial results: {error}</span>
					</div>
				{/if}

				<div class="grid-cell">
					{#if predictResult}
						<HeatmapOverlay
							gradcamBase64={predictResult.gradcam_base64}
							className={predictResult.class_name}
							confidence={predictResult.confidence}
						/>
					{:else}
						<div class="empty-cell">
							<span class="empty-label">Classification unavailable</span>
						</div>
					{/if}
				</div>

				<div class="grid-cell">
					{#if segResult}
						<SegmentationMask
							{originalSrc}
							maskBase64={segResult.mask_base64}
							onDefectArea={(pct) => (defectAreaPct = pct)}
						/>
					{:else}
						<div class="empty-cell">
							<span class="empty-label">Segmentation unavailable</span>
						</div>
					{/if}
				</div>

				<div class="grid-cell">
					{#if predictResult}
						<ConfidenceChart
							probabilities={predictResult.class_probabilities}
						/>
					{:else}
						<div class="empty-cell">
							<span class="empty-label">Probabilities unavailable</span>
						</div>
					{/if}
				</div>

				<div class="grid-cell">
					{#if predictResult}
						<ImageStatistics
							className={predictResult.class_name}
							confidence={predictResult.confidence}
							classProbabilities={predictResult.class_probabilities}
							{defectAreaPct}
						/>
					{:else}
						<div class="empty-cell">
							<span class="empty-label">Statistics unavailable</span>
						</div>
					{/if}
				</div>
			</div>
		{:else}
			<div class="state-screen">
				<div class="empty-icon">
					<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
						<rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
						<circle cx="8.5" cy="8.5" r="1.5" />
						<polyline points="21 15 16 10 5 21" />
					</svg>
				</div>
				<h2 class="state-title">Ready for Inspection</h2>
				<p class="state-desc">
					Upload a radiographic weld image to begin defect analysis.<br />
					The system will classify defects, generate a Grad-CAM heatmap,<br />
					and produce a segmentation mask.
				</p>
				<div class="feature-tags">
					<span class="feature-tag">Classification</span>
					<span class="feature-tag">Grad-CAM</span>
					<span class="feature-tag">Segmentation</span>
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	.page-layout {
		display: flex;
		height: 100%;
		overflow: hidden;
	}

	.sidebar {
		width: var(--sidebar-width);
		flex-shrink: 0;
		background: var(--color-bg-secondary);
		border-right: 1px solid var(--color-border);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.sidebar-content {
		flex: 1;
		display: flex;
		flex-direction: column;
		padding: var(--space-lg);
		gap: var(--space-lg);
		overflow-y: auto;
	}

	.sidebar-section {
		display: flex;
		flex-direction: column;
		gap: var(--space-md);
	}

	.section-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--color-text-muted);
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}

	.btn-reset {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-sm);
		width: 100%;
		padding: 10px var(--space-md);
		font-size: 13px;
		font-weight: 500;
		color: var(--color-text-secondary);
		background: var(--color-bg-tertiary);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-md);
		transition: all var(--transition-fast);
	}

	.btn-reset:hover {
		color: var(--color-text-primary);
		background: var(--color-bg-hover);
		border-color: var(--color-border-hover);
	}

	.sidebar-footer {
		margin-top: auto;
		padding-top: var(--space-md);
		border-top: 1px solid var(--color-border);
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--space-sm);
	}

	.footer-text {
		font-size: 11px;
		font-weight: 500;
		color: var(--color-text-muted);
		font-family: var(--font-mono);
	}

	.main-area {
		flex: 1;
		overflow: auto;
		background: var(--color-bg-primary);
	}

	.state-screen {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		gap: var(--space-md);
		padding: var(--space-2xl);
		text-align: center;
	}

	.state-title {
		font-size: 20px;
		font-weight: 600;
		color: var(--color-text-primary);
	}

	.state-desc {
		font-size: 13px;
		color: var(--color-text-muted);
		line-height: 1.6;
		max-width: 400px;
	}

	.spinner {
		width: 40px;
		height: 40px;
		border: 3px solid var(--color-border);
		border-top-color: var(--color-accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	.error-icon {
		color: var(--color-error);
	}

	.btn-retry {
		margin-top: var(--space-sm);
		padding: 10px 24px;
		font-size: 13px;
		font-weight: 600;
		color: var(--color-bg-primary);
		background: var(--color-accent);
		border-radius: var(--radius-md);
		transition: all var(--transition-fast);
	}

	.btn-retry:hover {
		background: var(--color-accent-hover);
	}

	.empty-icon {
		color: var(--color-text-muted);
		opacity: 0.4;
		margin-bottom: var(--space-sm);
	}

	.feature-tags {
		display: flex;
		gap: var(--space-sm);
		margin-top: var(--space-sm);
	}

	.feature-tag {
		padding: 4px 12px;
		font-size: 11px;
		font-weight: 500;
		font-family: var(--font-mono);
		color: var(--color-accent);
		background: var(--color-accent-dim);
		border: 1px solid var(--color-border-accent);
		border-radius: 20px;
	}

	.results-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		grid-template-rows: 1fr 1fr;
		gap: var(--space-md);
		padding: var(--space-md);
		height: 100%;
		min-height: 0;
	}

	.partial-error {
		grid-column: 1 / -1;
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-sm) var(--space-md);
		font-size: 12px;
		color: var(--color-warning);
		background: rgba(245, 158, 11, 0.1);
		border: 1px solid rgba(245, 158, 11, 0.3);
		border-radius: var(--radius-md);
	}

	.grid-cell {
		min-height: 0;
		min-width: 0;
		display: flex;
		flex-direction: column;
	}

	.grid-cell > :global(*) {
		flex: 1;
		min-height: 0;
	}

	.empty-cell {
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--color-bg-secondary);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-lg);
	}

	.empty-label {
		font-size: 13px;
		color: var(--color-text-muted);
	}
</style>
