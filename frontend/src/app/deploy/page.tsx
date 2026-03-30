"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { ServingStatus, Dataset } from "@/types";

export default function DeployPage() {
  const [status, setStatus] = useState<ServingStatus | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [jobs, setJobs] = useState<any[]>([]);
  const [deploying, setDeploying] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [featureInput, setFeatureInput] = useState("{}");
  const [prediction, setPrediction] = useState<any>(null);
  const [error, setError] = useState("");
  const [deployError, setDeployError] = useState("");

  useEffect(() => {
    loadStatus();
    api.listDatasets().then(setDatasets).catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      api.listJobs(selectedDataset).then((j) => {
        setJobs(j.filter((job: any) => job.status === "completed"));
      }).catch(() => {});
    }
  }, [selectedDataset]);

  async function loadStatus() {
    try {
      const s = await api.servingStatus();
      setStatus(s);
    } catch {}
  }

  async function handleDeploy(jobId: string) {
    setDeploying(true);
    setDeployError("");
    try {
      await api.deployModel(jobId);
      await loadStatus();
    } catch (e: any) {
      setDeployError(e.message);
    } finally {
      setDeploying(false);
    }
  }

  async function handlePredict() {
    setPredicting(true);
    setError("");
    setPrediction(null);
    try {
      const features = JSON.parse(featureInput);
      const result = await api.predict(features);
      setPrediction(result);
    } catch (e: any) {
      setError(e.message || "Invalid JSON or prediction failed");
    } finally {
      setPredicting(false);
    }
  }

  async function handleUndeploy() {
    try {
      await api.undeployModel();
      await loadStatus();
    } catch {}
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-1">Deploy & Predict</h1>
      <p className="text-gray-500 text-sm mb-6">
        Deploy a trained model and test predictions via the REST API
      </p>

      {/* Current Status */}
      <div className={`rounded-lg border p-5 mb-6 ${
        status?.status === "deployed"
          ? "bg-emerald-50 border-emerald-200"
          : "bg-gray-50 border-gray-200"
      }`}>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-semibold text-gray-900 flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${
                status?.status === "deployed" ? "bg-emerald-500 animate-pulse" : "bg-gray-300"
              }`} />
              Model Status: {status?.status === "deployed" ? "Active" : "No Model Deployed"}
            </h2>
            {status?.status === "deployed" && (
              <div className="mt-2 text-sm text-gray-600 space-y-0.5">
                <p>Model: <strong>{status.model_type}</strong></p>
                <p>Job ID: <code className="text-xs bg-white px-1 py-0.5 rounded">{status.job_id}</code></p>
                {status.metrics && (
                  <p>Metrics: {Object.entries(status.metrics)
                    .filter(([k]) => k !== "training_duration" && k !== "confusion_matrix")
                    .map(([k, v]) => `${k}=${(v as number).toFixed(4)}`)
                    .join(", ")}</p>
                )}
                {status.deployed_at && <p className="text-xs text-gray-400">Deployed: {status.deployed_at}</p>}
              </div>
            )}
          </div>
          {status?.status === "deployed" && (
            <button
              onClick={handleUndeploy}
              className="text-xs text-red-500 hover:text-red-700 border border-red-200 rounded px-2 py-1"
            >
              Undeploy
            </button>
          )}
        </div>
      </div>

      {/* Deploy a model */}
      <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Deploy a Model</h3>

        <div className="mb-4">
          <label className="text-xs text-gray-500 mb-1 block">Select Dataset</label>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm w-full"
          >
            <option value="">Choose a dataset...</option>
            {datasets.map((ds) => (
              <option key={ds.id} value={ds.id}>{ds.name}</option>
            ))}
          </select>
        </div>

        {jobs.length > 0 && (
          <div className="space-y-2">
            {jobs.map((job) => (
              <div
                key={job.id}
                className="flex items-center justify-between p-3 rounded-lg border border-gray-100 hover:border-gray-200"
              >
                <div>
                  <p className="text-sm font-medium text-gray-900">{job.model_type}</p>
                  <p className="text-xs text-gray-500">
                    {job.metrics
                      ? Object.entries(job.metrics)
                          .filter(([k]) => k !== "training_duration" && k !== "confusion_matrix")
                          .map(([k, v]) => `${k}: ${(v as number).toFixed(4)}`)
                          .join(" · ")
                      : "No metrics"}
                  </p>
                </div>
                <button
                  onClick={() => handleDeploy(job.id)}
                  disabled={deploying}
                  className="bg-blue-600 text-white text-xs font-medium px-3 py-1.5 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                >
                  {deploying ? "Deploying..." : "Deploy"}
                </button>
              </div>
            ))}
          </div>
        )}

        {selectedDataset && jobs.length === 0 && (
          <p className="text-sm text-gray-400">No completed training jobs for this dataset</p>
        )}

        {deployError && <p className="text-sm text-red-600 mt-3">{deployError}</p>}
      </div>

      {/* Test Predictions */}
      {status?.status === "deployed" && (
        <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Test Prediction</h3>
          <p className="text-xs text-gray-500 mb-3">
            Enter feature values as JSON. Keys must match the training feature names.
          </p>

          <textarea
            value={featureInput}
            onChange={(e) => setFeatureInput(e.target.value)}
            rows={6}
            className="w-full border border-gray-200 rounded-lg px-4 py-3 text-sm font-mono focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-100 mb-3"
            placeholder='{"feature1": 1.0, "feature2": "value", ...}'
          />

          <div className="flex items-center gap-3">
            <button
              onClick={handlePredict}
              disabled={predicting}
              className="bg-emerald-600 text-white text-sm font-medium px-5 py-2 rounded-lg hover:bg-emerald-700 disabled:opacity-50"
            >
              {predicting ? "Predicting..." : "Get Prediction"}
            </button>
            <button
              onClick={() => setFeatureInput("{}")}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear
            </button>
          </div>

          {error && <p className="text-sm text-red-600 mt-3">{error}</p>}

          {prediction && (
            <div className="mt-4 bg-emerald-50 rounded-lg border border-emerald-200 p-4">
              <h4 className="text-sm font-semibold text-emerald-800 mb-2">Prediction Result</h4>
              <div className="font-mono text-sm text-emerald-900">
                {(() => {
                  const ds = datasets.find((d) => d.id === selectedDataset);
                  const targetName = ds?.target_column;
                  return targetName ? (
                    <p>Target: <strong>{targetName}</strong> = <strong>{JSON.stringify(prediction.prediction)}</strong></p>
                  ) : (
                    <p>Prediction: <strong>{JSON.stringify(prediction.prediction)}</strong></p>
                  );
                })()}
                {prediction.probabilities && (
                  <p className="mt-1">
                    Probabilities: {prediction.probabilities.map(
                      (probs: number[]) => probs.map((p) => p.toFixed(4)).join(", ")
                    ).join(" | ")}
                  </p>
                )}
                <p className="text-xs text-emerald-600 mt-1">Model: {prediction.model_type}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* API Documentation */}
      <div className="bg-white rounded-lg border border-gray-200 p-5">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">API Endpoints</h3>
        <div className="space-y-3 text-sm">
          <div>
            <code className="text-xs bg-gray-100 px-2 py-1 rounded font-medium">
              POST /api/serving/predict
            </code>
            <p className="text-xs text-gray-500 mt-1">Single prediction. Body: {"{ \"features\": { ... } }"}</p>
          </div>
          <div>
            <code className="text-xs bg-gray-100 px-2 py-1 rounded font-medium">
              POST /api/serving/predict/batch
            </code>
            <p className="text-xs text-gray-500 mt-1">Batch predictions. Body: {"{ \"records\": [{ ... }, ...] }"}</p>
          </div>
          <div>
            <code className="text-xs bg-gray-100 px-2 py-1 rounded font-medium">
              GET /api/serving/status
            </code>
            <p className="text-xs text-gray-500 mt-1">Check current deployment status</p>
          </div>
          <div className="pt-2 border-t border-gray-100">
            <p className="text-xs text-gray-400">
              Full API docs available at: <code>/docs</code> (Swagger UI)
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
