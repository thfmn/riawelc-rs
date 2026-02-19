<script lang="ts">
	interface Props {
		originalSrc: string;
		maskBase64: string;
		onDefectArea?: (pct: number) => void;
	}

	let { originalSrc, maskBase64, onDefectArea }: Props = $props();

	let showOverlay = $state(true);
	let canvasEl: HTMLCanvasElement | undefined = $state();

	// Cache loaded image elements so we don't re-fetch on toggle
	let origImg: HTMLImageElement | undefined = $state();
	let maskImg: HTMLImageElement | undefined = $state();

	// Load images when props change
	$effect(() => {
		const oi = new Image();
		const mi = new Image();
		let loaded = 0;

		function onLoad() {
			loaded++;
			if (loaded < 2) return;
			origImg = oi;
			maskImg = mi;
		}

		oi.onload = onLoad;
		mi.onload = onLoad;
		oi.crossOrigin = 'anonymous';
		oi.src = originalSrc;
		mi.src = `data:image/png;base64,${maskBase64}`;
	});

	// Render canvas when images are loaded or overlay is toggled
	$effect(() => {
		if (!canvasEl || !origImg || !maskImg) return;

		const w = origImg.naturalWidth;
		const h = origImg.naturalHeight;
		canvasEl.width = w;
		canvasEl.height = h;

		const ctx = canvasEl.getContext('2d')!;

		// Draw original image
		ctx.drawImage(origImg, 0, 0, w, h);

		if (!showOverlay) return;

		const origData = ctx.getImageData(0, 0, w, h);

		// Draw mask to read pixel data
		ctx.drawImage(maskImg, 0, 0, w, h);
		const maskData = ctx.getImageData(0, 0, w, h);

		// Blend: where mask > 128, apply red at alpha 0.3
		const out = origData.data;
		const mask = maskData.data;
		const totalPixels = w * h;
		let defectPixels = 0;
		const alpha = 0.3;

		for (let i = 0; i < totalPixels; i++) {
			const idx = i * 4;
			if (mask[idx] > 128) {
				defectPixels++;
				out[idx] = Math.round(out[idx] * (1 - alpha) + 255 * alpha);
				out[idx + 1] = Math.round(out[idx + 1] * (1 - alpha));
				out[idx + 2] = Math.round(out[idx + 2] * (1 - alpha));
			}
		}

		ctx.putImageData(origData, 0, 0);

		const pct = (defectPixels / totalPixels) * 100;
		onDefectArea?.(pct);
	});
</script>

<div class="seg-panel">
	<div class="panel-header">
		<h3 class="panel-title">
			<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
				<rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
				<circle cx="8.5" cy="8.5" r="1.5" />
				<polyline points="21 15 16 10 5 21" />
			</svg>
			Segmentation Mask
		</h3>
		<label class="toggle-label" aria-label="Toggle mask overlay">
			<input type="checkbox" bind:checked={showOverlay} />
			<span class="toggle-text">Overlay</span>
		</label>
	</div>

	<div class="image-area">
		<canvas bind:this={canvasEl}></canvas>
	</div>
</div>

<style>
	.seg-panel {
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

	.toggle-label {
		display: flex;
		align-items: center;
		gap: 6px;
		cursor: pointer;
		font-size: 12px;
	}

	.toggle-label input[type='checkbox'] {
		appearance: none;
		width: 32px;
		height: 18px;
		background: var(--color-bg-hover);
		border-radius: 9px;
		position: relative;
		cursor: pointer;
		transition: background var(--transition-fast);
	}

	.toggle-label input[type='checkbox']::after {
		content: '';
		position: absolute;
		top: 2px;
		left: 2px;
		width: 14px;
		height: 14px;
		background: var(--color-text-muted);
		border-radius: 50%;
		transition: all var(--transition-fast);
	}

	.toggle-label input[type='checkbox']:checked {
		background: var(--color-accent);
	}

	.toggle-label input[type='checkbox']:checked::after {
		left: 16px;
		background: var(--color-bg-primary);
	}

	.toggle-text {
		color: var(--color-text-secondary);
		font-weight: 500;
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

	canvas {
		max-width: 100%;
		max-height: 100%;
		object-fit: contain;
		border-radius: var(--radius-md);
	}
</style>
