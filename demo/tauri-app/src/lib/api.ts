const API_BASE = 'http://localhost:8000';
const REQUEST_TIMEOUT = 60_000;

export interface PredictResponse {
	class_name: string;
	confidence: number;
	class_probabilities: Record<string, number>;
	gradcam_base64: string;
}

export interface UNetSegmentResponse {
	mask_base64: string;
	model_name: string;
}

export interface HealthResponse {
	status: string;
	version: string;
}

export interface ModelInfoResponse {
	model_name: string;
	input_shape: number[];
	num_classes: number;
	description: string;
}

export class ApiError extends Error {
	status?: number;

	constructor(message: string, status?: number) {
		super(message);
		this.name = 'ApiError';
		this.status = status;
	}
}

export function isApiError(err: unknown): err is ApiError {
	return err instanceof ApiError;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
	const controller = new AbortController();
	const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);
	try {
		const response = await fetch(url, { ...init, signal: controller.signal });
		if (!response.ok) {
			let message = `Server responded with ${response.status}`;
			try {
				const body = await response.json();
				if (body.detail) message = body.detail;
			} catch {
			}
			throw new ApiError(message, response.status);
		}
		return response.json() as Promise<T>;
	} catch (err: unknown) {
		if (isApiError(err)) throw err;
		if (err instanceof DOMException && err.name === 'AbortError') {
			throw new ApiError('Request timed out. Is the API server running?');
		}
		throw new ApiError('Cannot reach the API server. Make sure it is running at ' + API_BASE);
	} finally {
		clearTimeout(timeoutId);
	}
}

async function postFile<T>(endpoint: string, file: File): Promise<T> {
	const formData = new FormData();
	formData.append('file', file);
	return request<T>(`${API_BASE}${endpoint}`, { method: 'POST', body: formData });
}

export const predict = (file: File) => postFile<PredictResponse>('/predict', file);

export const segmentUnet = (file: File) =>
	postFile<UNetSegmentResponse>('/segment/unet/baseline', file);

export const checkHealth = () => request<HealthResponse>(`${API_BASE}/health`);

export const getModelInfo = () => request<ModelInfoResponse>(`${API_BASE}/model/info`);
