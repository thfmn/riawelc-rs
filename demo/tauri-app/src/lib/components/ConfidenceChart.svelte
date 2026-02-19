<script lang="ts">
	interface Props {
		probabilities: Record<string, number>;
	}

	let { probabilities }: Props = $props();

	let chartData = $derived.by(() => {
		const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
		const max = Math.max(...Object.values(probabilities), 0.01);
		return { entries, max };
	});

	function formatLabel(label: string): string {
		return label
			.replace(/_/g, ' ')
			.replace(/\b\w/g, (c) => c.toUpperCase());
	}
</script>

<div class="chart-panel">
	<div class="panel-header">
		<h3 class="panel-title">
			<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
				<line x1="18" y1="20" x2="18" y2="10" />
				<line x1="12" y1="20" x2="12" y2="4" />
				<line x1="6" y1="20" x2="6" y2="14" />
			</svg>
			Class Probabilities
		</h3>
	</div>

	<div class="chart-body">
		{#each chartData.entries as [className, probability], i}
			{@const pct = (probability * 100)}
			{@const barWidth = (probability / chartData.max) * 100}

			<div class="bar-row" style="animation-delay: {i * 60}ms">
				<div class="bar-label">
					<span class="class-name">{formatLabel(className)}</span>
					<span class="class-pct">{pct.toFixed(1)}%</span>
				</div>
				<div class="bar-track">
					<div
						class="bar-fill"
						style="width: {barWidth}%"
					></div>
				</div>
			</div>
		{/each}
	</div>
</div>

<style>
	.chart-panel {
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

	.chart-body {
		flex: 1;
		padding: var(--space-md);
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		gap: var(--space-sm);
	}

	.bar-row {
		animation: fadeIn 0.4s ease forwards;
		opacity: 0;
	}

	.bar-label {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 4px;
	}

	.class-name {
		font-size: 12px;
		font-weight: 500;
		color: var(--color-text-secondary);
	}

	.class-pct {
		font-size: 12px;
		font-family: var(--font-mono);
		font-weight: 500;
		color: var(--color-text-muted);
	}

	.bar-track {
		width: 100%;
		height: 8px;
		background: var(--color-bg-hover);
		border-radius: 4px;
		overflow: hidden;
	}

	.bar-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--color-accent), var(--color-accent-hover));
		border-radius: 4px;
		transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
	}

	@keyframes fadeIn {
		from {
			opacity: 0;
			transform: translateX(-8px);
		}
		to {
			opacity: 1;
			transform: translateX(0);
		}
	}
</style>
