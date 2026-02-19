<script lang="ts">
	import type { Snippet } from 'svelte';
	import { onMount } from 'svelte';
	import { checkHealth } from '$lib/api';
	import '../app.css';

	let { children }: { children: Snippet } = $props();

	let apiOnline = $state(false);
	let checking = $state(true);

	async function pollHealth() {
		try {
			await checkHealth();
			apiOnline = true;
		} catch {
			apiOnline = false;
		} finally {
			checking = false;
		}
	}

	onMount(() => {
		pollHealth();
		const interval = setInterval(pollHealth, 15_000);
		return () => clearInterval(interval);
	});
</script>

<div class="app-shell">
	<header class="app-header">
		<div class="header-brand">
			<div class="brand-icon">
				<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
					<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" />
					<path d="M12 8v4l3 3" />
				</svg>
			</div>
			<div class="brand-text">
				<h1 class="brand-name">RIAWELC Inspector</h1>
				<span class="brand-sub">Welding Defect Analysis</span>
			</div>
		</div>

		<div class="header-meta">
			<span class="header-badge">Demo</span>
			<span class="api-status" class:online={apiOnline} class:offline={!apiOnline && !checking}>
				<span class="status-dot"></span>
				{#if checking}
					Connecting...
				{:else if apiOnline}
					API Online
				{:else}
					API Offline
				{/if}
			</span>
		</div>
	</header>

	<main class="app-body">
		{@render children()}
	</main>
</div>

<style>
	.app-shell {
		display: flex;
		flex-direction: column;
		height: 100vh;
		overflow: hidden;
	}

	.app-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		height: var(--header-height);
		padding: 0 var(--space-lg);
		background: var(--color-bg-secondary);
		border-bottom: 1px solid var(--color-border);
		flex-shrink: 0;
		-webkit-app-region: drag;
	}

	.header-brand {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		-webkit-app-region: no-drag;
	}

	.brand-icon {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 36px;
		height: 36px;
		background: var(--color-accent-dim);
		border: 1px solid var(--color-border-accent);
		border-radius: var(--radius-md);
		color: var(--color-accent);
	}

	.brand-text {
		display: flex;
		flex-direction: column;
	}

	.brand-name {
		font-size: 16px;
		font-weight: 700;
		color: var(--color-text-primary);
		letter-spacing: -0.02em;
	}

	.brand-sub {
		font-size: 11px;
		color: var(--color-text-muted);
		font-weight: 500;
		letter-spacing: 0.02em;
	}

	.header-meta {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		-webkit-app-region: no-drag;
	}

	.header-badge {
		padding: 3px 10px;
		font-size: 10px;
		font-weight: 600;
		font-family: var(--font-mono);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--color-text-muted);
		background: var(--color-bg-tertiary);
		border: 1px solid var(--color-border);
		border-radius: 20px;
	}

	.api-status {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 3px 10px;
		font-size: 10px;
		font-weight: 600;
		font-family: var(--font-mono);
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--color-text-muted);
		background: var(--color-bg-tertiary);
		border: 1px solid var(--color-border);
		border-radius: 20px;
	}

	.api-status.online {
		color: var(--color-success);
		border-color: rgba(34, 197, 94, 0.3);
	}

	.api-status.offline {
		color: var(--color-error);
		border-color: rgba(239, 68, 68, 0.3);
	}

	.status-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--color-text-muted);
		flex-shrink: 0;
	}

	.api-status.online .status-dot {
		background: var(--color-success);
		box-shadow: 0 0 6px rgba(34, 197, 94, 0.5);
	}

	.api-status.offline .status-dot {
		background: var(--color-error);
		box-shadow: 0 0 6px rgba(239, 68, 68, 0.5);
	}

	.app-body {
		flex: 1;
		overflow: hidden;
	}
</style>
