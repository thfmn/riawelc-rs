<script lang="ts">
	import { getModelInfo, checkHealth, isApiError, type ModelInfoResponse, type HealthResponse } from '$lib/api';
	import { onMount } from 'svelte';

	let modelInfo: ModelInfoResponse | null = $state(null);
	let health: HealthResponse | null = $state(null);
	let error: string | null = $state(null);
	let loading = $state(true);

	onMount(async () => {
		await fetchInfo();
	});

	async function fetchInfo() {
		loading = true;
		error = null;
		try {
			const [h, m] = await Promise.all([checkHealth(), getModelInfo()]);
			health = h;
			modelInfo = m;
		} catch (err: unknown) {
			error = isApiError(err) ? err.message : 'Failed to connect to API';
		} finally {
			loading = false;
		}
	}
</script>

<div class="model-info">
	<div class="section-header">
		<h3 class="section-title">
			<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
				<path d="M12 2L2 7l10 5 10-5-10-5z" />
				<path d="M2 17l10 5 10-5" />
				<path d="M2 12l10 5 10-5" />
			</svg>
			Model Info
		</h3>
		<button class="refresh-btn" onclick={fetchInfo} title="Refresh" disabled={loading} aria-label="Refresh model info">
			<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class:spinning={loading} aria-hidden="true">
				<polyline points="23 4 23 10 17 10" />
				<path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
			</svg>
		</button>
	</div>

	{#if loading}
		<div class="status-row">
			<div class="spinner-small"></div>
			<span class="status-text">Connecting...</span>
		</div>
	{:else if error}
		<div class="status-row error">
			<div class="status-dot offline"></div>
			<span class="status-text">{error}</span>
		</div>
	{:else}
		<div class="status-row">
			<div class="status-dot online"></div>
			<span class="status-text">
				API Online
				{#if health?.version}
					<span class="version">v{health.version}</span>
				{/if}
			</span>
		</div>

		{#if modelInfo}
			<div class="info-grid">
				<div class="info-item">
					<span class="info-label">Model</span>
					<span class="info-value">{modelInfo.model_name}</span>
				</div>
				<div class="info-item">
					<span class="info-label">Input</span>
					<span class="info-value mono">{modelInfo.input_shape.join(' x ')}</span>
				</div>
				<div class="info-item">
					<span class="info-label">Classes</span>
					<span class="info-value mono">{modelInfo.num_classes}</span>
				</div>
			</div>

			{#if modelInfo.description}
				<p class="description">{modelInfo.description}</p>
			{/if}
		{/if}
	{/if}
</div>

<style>
	.model-info {
		background: var(--color-bg-tertiary);
		border: 1px solid var(--color-border);
		border-radius: var(--radius-lg);
		padding: var(--space-md);
		display: flex;
		flex-direction: column;
		gap: var(--space-sm);
	}

	.section-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.section-title {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		font-size: 13px;
		font-weight: 600;
		color: var(--color-text-secondary);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.section-title svg {
		color: var(--color-accent);
	}

	.refresh-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 28px;
		height: 28px;
		background: transparent;
		border-radius: var(--radius-sm);
		color: var(--color-text-muted);
		transition: all var(--transition-fast);
	}

	.refresh-btn:hover {
		background: var(--color-bg-hover);
		color: var(--color-accent);
	}

	.spinning {
		animation: spin 1s linear infinite;
	}

	.status-row {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-sm) 0;
	}

	.status-row.error .status-text {
		color: var(--color-error);
		font-size: 11px;
	}

	.status-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.status-dot.online {
		background: var(--color-success);
		box-shadow: 0 0 6px rgba(34, 197, 94, 0.5);
	}

	.status-dot.offline {
		background: var(--color-error);
		box-shadow: 0 0 6px rgba(239, 68, 68, 0.5);
	}

	.status-text {
		font-size: 12px;
		color: var(--color-text-secondary);
		font-weight: 500;
	}

	.version {
		font-family: var(--font-mono);
		font-size: 11px;
		color: var(--color-text-muted);
		margin-left: 4px;
	}

	.spinner-small {
		width: 14px;
		height: 14px;
		border: 2px solid var(--color-border);
		border-top-color: var(--color-accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	.info-grid {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.info-item {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 4px 0;
	}

	.info-label {
		font-size: 12px;
		color: var(--color-text-muted);
		font-weight: 500;
	}

	.info-value {
		font-size: 12px;
		color: var(--color-text-primary);
		font-weight: 500;
	}

	.info-value.mono {
		font-family: var(--font-mono);
	}

	.description {
		font-size: 11px;
		color: var(--color-text-muted);
		line-height: 1.5;
		padding-top: var(--space-sm);
		border-top: 1px solid var(--color-border);
	}
</style>
