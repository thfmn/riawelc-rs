<script lang="ts">
	interface Props {
		className: string;
		confidence: number;
		classProbabilities: Record<string, number>;
		defectAreaPct: number | null;
	}

	let { className, confidence, classProbabilities, defectAreaPct }: Props = $props();

	const NON_DEFECT_CLASSES = new Set(['good', 'no_defect']);

	let isDefect = $derived(!NON_DEFECT_CLASSES.has(className.toLowerCase()));

	let sorted = $derived(
		Object.entries(classProbabilities).sort((a, b) => b[1] - a[1])
	);

	let runnerUp = $derived(sorted.length > 1 ? sorted[1] : null);

	let confidenceMargin = $derived(
		runnerUp ? confidence - runnerUp[1] : confidence
	);

	let interpretation = $derived.by(() => {
		if (confidence >= 0.9) return { text: 'High confidence', level: 'high' };
		if (confidence >= 0.7) return { text: 'Review recommended', level: 'medium' };
		return { text: 'Manual inspection advised', level: 'low' };
	});

	function formatLabel(label: string): string {
		return label
			.replace(/_/g, ' ')
			.replace(/\b\w/g, (c) => c.toUpperCase());
	}
</script>

<div class="stats-panel">
	<div class="panel-header">
		<h3 class="panel-title">
			<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
				<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
				<polyline points="14 2 14 8 20 8" />
				<line x1="16" y1="13" x2="8" y2="13" />
				<line x1="16" y1="17" x2="8" y2="17" />
			</svg>
			Image Statistics
		</h3>
	</div>

	<div class="stats-body">
		<div class="stat-row headline">
			<span class="stat-label">Prediction</span>
			<span class="stat-value">
				<span class="class-badge" class:defect={isDefect}>
					{formatLabel(className)}
				</span>
				<span class="conf-value">{(confidence * 100).toFixed(1)}%</span>
			</span>
		</div>

		<div class="stat-row">
			<span class="stat-label">Interpretation</span>
			<span class="stat-value interp" class:high={interpretation.level === 'high'} class:medium={interpretation.level === 'medium'} class:low={interpretation.level === 'low'}>
				{interpretation.text}
			</span>
		</div>

		{#if defectAreaPct !== null}
			<div class="stat-row">
				<span class="stat-label">Defect area</span>
				<span class="stat-value mono">{defectAreaPct.toFixed(2)}%</span>
			</div>
		{/if}

		<div class="stat-row">
			<span class="stat-label">Confidence margin</span>
			<span class="stat-value mono">{(confidenceMargin * 100).toFixed(1)}pp</span>
		</div>

		{#if runnerUp}
			<div class="stat-row">
				<span class="stat-label">Runner-up</span>
				<span class="stat-value">
					{formatLabel(runnerUp[0])}
					<span class="secondary">({(runnerUp[1] * 100).toFixed(1)}%)</span>
				</span>
			</div>
		{/if}

	</div>
</div>

<style>
	.stats-panel {
		background: var(--color-bg-secondary);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-lg);
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}

	.panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: var(--space-md);
		border-bottom: 1px solid var(--color-border);
	}

	.panel-title {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		font-size: 14px;
		font-weight: 600;
		color: var(--color-text-primary);
	}

	.panel-title svg {
		color: var(--color-accent);
	}

	.stats-body {
		flex: 1;
		padding: var(--space-md);
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: var(--space-sm);
	}

	.stat-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 6px 0;
	}

	.stat-row.headline {
		padding-bottom: var(--space-sm);
		border-bottom: 1px solid var(--color-border);
		margin-bottom: 2px;
	}

	.stat-label {
		font-size: 12px;
		font-weight: 500;
		color: var(--color-text-muted);
	}

	.stat-value {
		font-size: 13px;
		font-weight: 500;
		color: var(--color-text-primary);
		display: flex;
		align-items: center;
		gap: var(--space-sm);
	}

	.stat-value.mono {
		font-family: var(--font-mono);
	}

	.class-badge {
		padding: 2px 10px;
		border-radius: 12px;
		font-size: 12px;
		font-weight: 600;
		font-family: var(--font-mono);
		background: rgba(34, 197, 94, 0.15);
		color: var(--color-success);
		border: 1px solid rgba(34, 197, 94, 0.3);
	}

	.class-badge.defect {
		background: rgba(239, 68, 68, 0.15);
		color: var(--color-error);
		border-color: rgba(239, 68, 68, 0.3);
	}

	.conf-value {
		font-size: 13px;
		font-family: var(--font-mono);
		font-weight: 600;
		color: var(--color-text-secondary);
	}

	.interp {
		font-weight: 600;
		font-size: 12px;
	}

	.interp.high {
		color: var(--color-success);
	}

	.interp.medium {
		color: var(--color-warning, #f59e0b);
	}

	.interp.low {
		color: var(--color-error);
	}

	.secondary {
		font-size: 11px;
		color: var(--color-text-muted);
		font-family: var(--font-mono);
	}

</style>
