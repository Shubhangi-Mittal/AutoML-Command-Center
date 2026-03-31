const API_URL = process.env.NEXT_PUBLIC_API_URL || "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_URL}${path}`;
  const headers = new Headers(options?.headers);

  // Avoid forcing CORS preflight on simple GET requests.
  if (options?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(url, {
    headers,
    ...options,
  });
  if (!res.ok) {
    const err = await res
      .json()
      .catch(async () => ({ detail: (await res.text().catch(() => "")) || res.statusText }));
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
  getDatasetVersions: (id: string) => request<any>(`/api/datasets/${id}/versions`),

  // Training
  launchTraining: (body: {
    dataset_id: string;
    model_types?: string[];
    optimization_metric?: string;
    target_column?: string;
    task_type?: string;
    cv_folds?: number;
    tune_hyperparameters?: boolean;
  }) =>
    request<any>("/api/training/launch", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listJobs: (datasetId?: string) =>
    request<any[]>(
      `/api/training/${datasetId ? `?dataset_id=${datasetId}` : ""}`
    ),

  getJob: (id: string) => request<any>(`/api/training/${id}`),

  // Experiments
  listExperiments: (datasetId?: string) =>
    request<any[]>(
      `/api/experiments/${datasetId ? `?dataset_id=${datasetId}` : ""}`
    ),

  getExperiment: (id: string) => request<any>(`/api/experiments/${id}`),

  compareModels: (experimentId: string) =>
    request<any>(`/api/experiments/${experimentId}/compare`),
  updateExperimentMetadata: (
    experimentId: string,
    body: {
      name?: string;
      tags?: string[];
      favorite?: boolean;
      archived?: boolean;
      notes?: string;
    }
  ) =>
    request<any>(`/api/experiments/${experimentId}/metadata`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  getExperimentReport: (experimentId: string) =>
    request<any>(`/api/experiments/${experimentId}/report`),

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
      body: JSON.stringify({ session_id: sessionId || "default" }),
    }),

  // Serving
  deployModel: (jobId: string) =>
    request("/api/serving/deploy", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId }),
    }),

  undeployModel: () =>
    request("/api/serving/undeploy", { method: "POST" }),

  predict: (features: Record<string, any>) =>
    request("/api/serving/predict", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),
  explainPrediction: (features: Record<string, any>, topK = 5) =>
    request<any>("/api/serving/explain", {
      method: "POST",
      body: JSON.stringify({ features, top_k: topK }),
    }),

  servingStatus: () => request<any>("/api/serving/status"),
  getPredictionTemplate: (datasetId: string) =>
    request<any>(`/api/serving/template/${datasetId}`),
  predictionHistory: (datasetId?: string) =>
    request<any>(`/api/serving/history${datasetId ? `?dataset_id=${datasetId}` : ""}`),

  // Health
  health: () => request<any>("/health"),
};
