"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import {
  Dataset,
  PredictionExplanation,
  PredictionHistoryItem,
  PredictionTemplate,
  ServingStatus,
  TrainingJob,
} from "@/types";

export default function DeployPage() {
  const [status, setStatus] = useState<ServingStatus | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>("");
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [deploying, setDeploying] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [featureInput, setFeatureInput] = useState("{}");
  const [prediction, setPrediction] = useState<any>(null);
  const [template, setTemplate] = useState<PredictionTemplate | null>(null);
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null);
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [error, setError] = useState("");
  const [deployError, setDeployError] = useState("");

  useEffect(() => {
    loadStatus();
    api.listDatasets().then(setDatasets).catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      api.listJobs(selectedDataset).then((j) => {
        setJobs(j.filter((job: TrainingJob) => job.status === "completed"));
      }).catch(() => {});
      api.getPredictionTemplate(selectedDataset).then(setTemplate).catch(() => setTemplate(null));
      api.predictionHistory(selectedDataset).then((result) => setHistory(result.predictions || [])).catch(() => setHistory([]));
    } else {
      setJobs([]);
      setTemplate(null);
      setHistory([]);
    }
  }, [selectedDataset]);

  async function loadStatus() {
    try {
      const s = await api.servingStatus();
      setStatus(s);
      if (s?.dataset_id) {
        setSelectedDataset(s.dataset_id);
      }
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
      const explain = await api.explainPrediction(features, 5).catch(() => null);
      setExplanation(explain);
      if (selectedDataset) {
        api.predictionHistory(selectedDataset).then((response) => setHistory(response.predictions || [])).catch(() => {});
      }
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

  function loadSampleInput() {
    if (!template) return;
    setFeatureInput(JSON.stringify(template.sample_input, null, 2));
    setError("");
  }

  function downloadSampleInput() {
    if (!template) return;
    const blob = new Blob([JSON.stringify(template.sample_input, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${template.dataset_name}-sample-input.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="relative overflow-hidden rounded-[2rem] border border-slate-200/80 bg-[radial-gradient(circle_at_top_left,_rgba(16,185,129,0.16),_transparent_24%),radial-gradient(circle_at_bottom_right,_rgba(14,165,233,0.12),_transparent_18%),linear-gradient(135deg,_rgba(255,255,255,0.95),_rgba(247,250,252,0.9))] px-8 py-10 mb-8 shadow-[0_24px_70px_rgba(15,23,42,0.07)]">
        <p className="text-xs font-semibold uppercase tracking-[0.28em] text-emerald-600 mb-3">Serving Console</p>
        <h1 className="font-display text-4xl font-bold text-gray-900 mb-3">Deploy a model, probe predictions, and watch inference activity.</h1>
        <p className="text-slate-600 text-sm md:text-base max-w-2xl leading-7">
          Move from experiment results to a live prediction workflow with sample payloads, explanation traces, and lightweight serving telemetry.
        </p>
      </div>

      {/* Current Status */}
      <div className={`glass-card rounded-[1.75rem] p-6 mb-6 ${
        status?.status === "deployed"
          ? "bg-emerald-50/80 border-emerald-200"
          : "bg-white/70 border-gray-200"
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
                {status.dataset_name && <p>Dataset: <strong>{status.dataset_name}</strong></p>}
                {status.target_column && <p>Target: <strong>{status.target_column}</strong></p>}
                <p>Job ID: <code className="text-xs bg-white px-1 py-0.5 rounded">{status.job_id}</code></p>
                {status.metrics && (
                  <p>Metrics: {Object.entries(status.metrics)
                    .filter(([k]) => k !== "training_duration" && k !== "confusion_matrix")
                    .map(([k, v]) => `${k}=${(v as number).toFixed(4)}`)
                    .join(", ")}</p>
                )}
                <p>Predictions served: <strong>{status.prediction_count ?? 0}</strong></p>
                <p>Average latency: <strong>{status.avg_latency_ms?.toFixed?.(2) ?? status.avg_latency_ms ?? 0} ms</strong></p>
                {status.last_prediction_at && <p>Last prediction: <strong>{status.last_prediction_at}</strong></p>}
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
      <div className="glass-card rounded-[1.75rem] p-6 mb-6">
        <h3 className="font-display text-xl font-semibold text-gray-800 mb-4">Deploy a Model</h3>

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
                className="flex items-center justify-between p-4 rounded-2xl border border-gray-100 hover:border-sky-200 bg-white/80"
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
        <div className="glass-card rounded-[1.75rem] p-6 mb-6">
          <h3 className="font-display text-xl font-semibold text-gray-800 mb-3">Test Prediction</h3>
          <p className="text-xs text-gray-500 mb-3">
            Enter feature values as JSON. Keys must match the training feature names.
          </p>

          {template && (
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-sky-100 bg-sky-50 px-4 py-3">
              <p className="text-xs text-blue-700">
                Sample payload ready for {template.dataset_name}
              </p>
              <button
                onClick={loadSampleInput}
                className="text-xs font-medium text-blue-700 hover:text-blue-900"
              >
                Load Sample JSON
              </button>
              <button
                onClick={downloadSampleInput}
                className="text-xs font-medium text-blue-700 hover:text-blue-900"
              >
                Download JSON
              </button>
            </div>
          )}

          <textarea
            value={featureInput}
            onChange={(e) => setFeatureInput(e.target.value)}
            rows={6}
            className="w-full border border-gray-200 rounded-2xl px-4 py-3 text-sm font-mono focus:outline-none focus:border-sky-400 focus:ring-1 focus:ring-sky-100 mb-3 bg-white/80"
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
            <div className="mt-4 rounded-[1.5rem] border border-emerald-200 bg-emerald-50 p-5">
              <h4 className="text-sm font-semibold text-emerald-800 mb-2">Prediction Result</h4>
              <div className="font-mono text-sm text-emerald-900">
                {(() => {
                  const ds = datasets.find((d) => d.id === selectedDataset);
                  const targetName = prediction.target_column || ds?.target_column;
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

          {explanation && explanation.top_contributors.length > 0 && (
            <div className="mt-4 rounded-[1.5rem] border border-gray-200 bg-white/90 p-5">
              <h4 className="text-sm font-semibold text-gray-800 mb-2">Prediction Explanation</h4>
              <div className="space-y-2">
                {explanation.top_contributors.map((item) => (
                  <div key={item.feature} className="flex items-center justify-between text-sm">
                    <span className="text-gray-700">{item.feature}</span>
                    <span className="font-mono text-gray-500">
                      impact={item.impact.toFixed(4)} · value={String(item.value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div className="glass-card rounded-[1.75rem] p-6 mb-6">
          <h3 className="font-display text-xl font-semibold text-gray-800 mb-3">Recent Prediction History</h3>
          <div className="space-y-2">
            {history.slice(0, 5).map((item, index) => (
              <div key={`${item.created_at}-${index}`} className="rounded-2xl border border-gray-100 bg-white/80 px-4 py-3 text-xs">
                <p className="text-gray-500">{item.created_at || "Unknown time"}</p>
                <p className="text-gray-700">Prediction: <strong>{JSON.stringify(item.prediction)}</strong></p>
                <p className="text-gray-500 truncate">Input: {JSON.stringify(item.features)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* API Documentation */}
      <div className="glass-card rounded-[1.75rem] p-6">
        <h3 className="font-display text-xl font-semibold text-gray-800 mb-3">API Endpoints</h3>
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
