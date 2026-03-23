const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  // Datasets
  uploadDataset: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API_URL}/api/datasets/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail);
    }
    return res.json();
  },

  listDatasets: () => request<any[]>("/api/datasets/"),

  getDataset: (id: string) => request<any>(`/api/datasets/${id}`),

  setTarget: (id: string, target_column: string, task_type: string) =>
    request(`/api/datasets/${id}/target`, {
      method: "PUT",
      body: JSON.stringify({ target_column, task_type }),
    }),

  deleteDataset: (id: string) =>
    request(`/api/datasets/${id}`, { method: "DELETE" }),

  // Training
  launchTraining: (body: {
    dataset_id: string;
    model_types: string[];
    optimization_metric: string;
  }) =>
    request<any>("/api/training/launch", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listJobs: (datasetId?: string) =>
    request<any[]>(
      `/api/training/jobs${datasetId ? `?dataset_id=${datasetId}` : ""}`
    ),

  getJob: (id: string) => request<any>(`/api/training/jobs/${id}`),

  // Experiments
  listExperiments: (datasetId?: string) =>
    request<any[]>(
      `/api/experiments/${datasetId ? `?dataset_id=${datasetId}` : ""}`
    ),

  getExperiment: (id: string) => request<any>(`/api/experiments/${id}`),

  compareModels: (experimentId: string) =>
    request<any>(`/api/experiments/${experimentId}/compare`),

  // Agent
  chat: (message: string, sessionId?: string, datasetId?: string) =>
    request<any>("/api/agent/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        session_id: sessionId || "default",
        dataset_id: datasetId,
      }),
    }),

  resetAgent: (sessionId?: string) =>
    request("/api/agent/reset", {
      method: "POST",
      body: JSON.stringify(sessionId || "default"),
    }),

  // Serving
  deployModel: (jobId: string) =>
    request("/api/serving/deploy", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId }),
    }),

  predict: (features: Record<string, any>) =>
    request("/api/serving/predict", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),

  servingStatus: () => request<any>("/api/serving/status"),

  // Health
  health: () => request<any>("/health"),

  // Available models
  availableModels: () => request<any>("/api/models/available"),
};
