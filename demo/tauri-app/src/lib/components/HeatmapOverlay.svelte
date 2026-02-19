<script lang="ts">
	interface Props {
		gradcamBase64: string;
		className: string;
		confidence: number;
	}

	let { gradcamBase64, className, confidence }: Props = $props();

	const NON_DEFECT_CLASSES = new Set(['good', 'no_defect']);
</script>

<div class="heatmap-panel">
	<div class="panel-header">
		<h3 class="panel-title">
			<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
				<circle cx="12" cy="12" r="10" />
				<path d="M12 16v-4" />
				<path d="M12 8h.01" />
			</svg>
			Grad-CAM Heatmap
		</h3>
		<div class="badge" class:defect={!NON_DEFECT_CLASSES.has(className.toLowerCase())}>
			{className}
			<span class="confidence-tag">{(confidence * 100).toFixed(1)}%</span>
		</div>
	</div>

	<div class="image-area">
		<img src="data:image/png;base64,{gradcamBase64}" alt="Grad-CAM heatmap overlay" />
	</div>
</div>

<style>
	.heatmap-panel {
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

	.badge {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: 4px 12px;
		border-radius: 20px;
		font-size: 12px;
		font-weight: 600;
		font-family: var(--font-mono);
		background: rgba(34, 197, 94, 0.15);
		color: var(--color-success);
		border: 1px solid rgba(34, 197, 94, 0.3);
	}

	.badge.defect {
		background: rgba(239, 68, 68, 0.15);
		color: var(--color-error);
		border-color: rgba(239, 68, 68, 0.3);
	}

	.confidence-tag {
		font-size: 11px;
		opacity: 0.8;
	}

	.image-area {
		flex: 1;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: var(--space-md);
		background: var(--color-bg-tertiary);
		min-height: 0;
		overflow: hidden;
	}

	.image-area img {
		max-width: 100%;
		max-height: 100%;
		object-fit: contain;
		border-radius: var(--radius-md);
	}
</style>
