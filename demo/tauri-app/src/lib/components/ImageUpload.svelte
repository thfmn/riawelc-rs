<script lang="ts">
    interface Props {
        onFileSelected: (file: File) => void;
        disabled?: boolean;
    }

    let { onFileSelected, disabled = false }: Props = $props();

    let dragOver = $state(false);
    let fileName = $state("");
    let validationError: string | null = $state(null);

    const ACCEPTED = ["image/png", "image/jpeg", "image/jpg"];

    function validateAndEmit(file: File | undefined) {
        if (!file) return;
        if (
            !ACCEPTED.includes(file.type) &&
            !file.name.match(/\.(png|jpe?g)$/i)
        ) {
            validationError =
                "Unsupported file type. Please upload a PNG or JPEG image.";
            return;
        }
        validationError = null;
        fileName = file.name;
        onFileSelected(file);
    }

    function handleDrop(e: DragEvent) {
        e.preventDefault();
        dragOver = false;
        if (disabled) return;
        const file = e.dataTransfer?.files[0];
        validateAndEmit(file);
    }

    function handleDragOver(e: DragEvent) {
        e.preventDefault();
        if (!disabled) dragOver = true;
    }

    function handleDragLeave() {
        dragOver = false;
    }

    function handleClick() {
        if (disabled) return;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ACCEPTED.join(",");
        input.onchange = () => {
            const file = input.files?.[0];
            validateAndEmit(file);
        };
        input.click();
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            handleClick();
        }
    }
</script>

<div
    class="upload-zone"
    class:drag-over={dragOver}
    class:disabled
    role="button"
    tabindex="0"
    aria-label="Upload radiograph image"
    ondrop={handleDrop}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    onclick={handleClick}
    onkeydown={handleKeydown}
>
    <div class="upload-icon">
        <svg
            width="40"
            height="40"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
        >
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
    </div>

    {#if fileName}
        <p class="file-name">{fileName}</p>
        <p class="hint">Drop or click to replace</p>
    {:else}
        <p class="label">Drop an image here</p>
        <p class="hint">or click to browse</p>
    {/if}

    <p class="formats">PNG, JPEG</p>
</div>

{#if validationError}
    <p class="validation-error" role="alert">{validationError}</p>
{/if}

<style>
    .upload-zone {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: var(--space-sm);
        padding: var(--space-xl) var(--space-md);
        border: 2px dashed var(--color-border);
        border-radius: var(--radius-lg);
        background: var(--color-bg-tertiary);
        cursor: pointer;
        transition: all var(--transition-normal);
        user-select: none;
    }

    .upload-zone:hover,
    .upload-zone:focus-visible {
        border-color: var(--color-accent);
        background: var(--color-accent-dim);
    }

    .upload-zone:focus-visible {
        outline: 2px solid var(--color-accent);
        outline-offset: 2px;
    }

    .drag-over {
        border-color: var(--color-accent);
        background: var(--color-accent-dim);
        box-shadow: var(--shadow-glow);
    }

    .disabled {
        opacity: 0.5;
        cursor: not-allowed;
        pointer-events: none;
    }

    .upload-icon {
        color: var(--color-text-muted);
        transition: color var(--transition-normal);
    }

    .upload-zone:hover .upload-icon,
    .drag-over .upload-icon {
        color: var(--color-accent);
    }

    .label {
        font-size: 15px;
        font-weight: 500;
        color: var(--color-text-primary);
    }

    .file-name {
        font-size: 13px;
        font-weight: 500;
        color: var(--color-accent);
        font-family: var(--font-mono);
        word-break: break-all;
        text-align: center;
        max-width: 100%;
    }

    .hint {
        font-size: 12px;
        color: var(--color-text-muted);
    }

    .formats {
        font-size: 11px;
        color: var(--color-text-muted);
        font-family: var(--font-mono);
        margin-top: var(--space-xs);
    }

    .validation-error {
        font-size: 12px;
        color: var(--color-error);
        padding: var(--space-sm) var(--space-md);
        margin-top: var(--space-sm);
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: var(--radius-md);
    }
</style>
