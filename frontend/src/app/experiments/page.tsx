"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { Dataset, Experiment, TrainResult, TrainingJob } from "@/types";

export default function ExperimentsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExp, setSelectedExp] = useState<any>(null);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [modelTypes, setModelTypes] = useState(["linear", "xgboost", "random_forest"]);
  const [metric, setMetric] = useState("f1");
  const [cvFolds, setCvFolds] = useState(1);
  const [tuneHyperparameters, setTuneHyperparameters] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      api.listExperiments(selectedDataset.id).then(setExperiments).catch(() => {});
    }
  }, [selectedDataset]);

  useEffect(() => {
    if (!selectedDataset?.task_type) return;
    setMetric(selectedDataset.task_type === "regression" ? "rmse" : "f1");
  }, [selectedDataset?.task_type]);

  async function handleTrain() {
    if (!selectedDataset?.target_column) {
      setError("Please set a target column first (Upload page → select dataset)");
      return;
    }
    setTraining(true);
    setError("");
    setTrainResult(null);
    try {
      const result = await api.launchTraining({
        dataset_id: selectedDataset.id,
        model_types: modelTypes,
        optimization_metric: metric,
        cv_folds: cvFolds,
        tune_hyperparameters: tuneHyperparameters,
      });
      setTrainResult(result);
      api.listExperiments(selectedDataset.id).then(setExperiments).catch(() => {});
      if (result.experiment_id) {
        loadExperiment(result.experiment_id);
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  }

  async function downloadReport(experimentId: string) {
    try {
      const result = await api.getExperimentReport(experimentId);
      const blob = new Blob([result.markdown], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `experiment-${experimentId}.md`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message || "Failed to export report");
    }
  }

  async function loadExperiment(expId: string) {
    try {
      const data = await api.compareModels(expId);
      setSelectedExp(data);
    } catch {}
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-1">Experiments</h1>
      <p className="text-gray-500 text-sm mb-6">Train models, compare results, and find the best performer</p>

      {/* Dataset selector */}
      <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Select Dataset</h3>
        <div className="flex flex-wrap gap-2">
          {datasets.map((ds) => (
            <button
              key={ds.id}
              onClick={() => { setSelectedDataset(ds); setTrainResult(null); setSelectedExp(null); }}
              className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                selectedDataset?.id === ds.id
                  ? "bg-blue-50 border-blue-300 text-blue-700 font-medium"
                  : "border-gray-200 text-gray-600 hover:border-gray-300"
              }`}
            >
              {ds.name}
            </button>
          ))}
          {datasets.length === 0 && (
            <p className="text-sm text-gray-400">No datasets uploaded yet</p>
          )}
        </div>
      </div>

      {!selectedDataset && (
        <div className="space-y-6 mb-6">
          {datasets.length > 0 ? (
            <>
              <div className="relative overflow-hidden rounded-3xl border border-slate-200 bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.18),_transparent_35%),radial-gradient(circle_at_bottom_right,_rgba(16,185,129,0.16),_transparent_30%),linear-gradient(135deg,_#f8fafc_0%,_#ffffff_55%,_#f0fdf4_100%)] p-8">
                <div className="max-w-3xl">
                  <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-600 mb-3">
                    Experiment Lab
                  </p>
                  <h2 className="text-3xl font-bold text-slate-900 mb-3">
                    Pick a dataset and turn it into a real model comparison.
                  </h2>
                  <p className="text-sm leading-6 text-slate-600 mb-6">
                    Launch baseline models, enable cross-validation, try lightweight tuning, then export a report once you find a winner.
                  </p>
                  <div className="flex flex-wrap gap-3">
                    {datasets.slice(0, 3).map((ds) => (
                      <button
                        key={ds.id}
                        onClick={() => { setSelectedDataset(ds); setTrainResult(null); setSelectedExp(null); }}
                        className="rounded-full border border-slate-300 bg-white/80 px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-sky-300 hover:text-sky-700"
                      >
                        Start with {ds.name}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-[1.3fr_0.7fr] gap-6">
                <div className="bg-white rounded-2xl border border-gray-200 p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-gray-800">Available Datasets</h3>
                    <span className="text-xs text-gray-400">{datasets.length} ready</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {datasets.map((ds) => (
                      <button
                        key={ds.id}
                        onClick={() => { setSelectedDataset(ds); setTrainResult(null); setSelectedExp(null); }}
                        className="rounded-2xl border border-gray-200 bg-white px-4 py-4 text-left transition hover:-translate-y-0.5 hover:border-sky-300 hover:shadow-md"
                      >
                        <div className="flex items-start justify-between gap-3 mb-3">
                          <div>
                            <p className="text-sm font-semibold text-gray-900">{ds.name}</p>
                            <p className="text-xs text-gray-500 mt-1">
                              {ds.rows?.toLocaleString() || "?"} rows x {ds.columns || "?"} columns
                            </p>
                          </div>
                          <span className={`rounded-full px-2 py-1 text-[11px] font-medium ${
                            ds.task_type === "regression"
                              ? "bg-amber-100 text-amber-700"
                              : "bg-emerald-100 text-emerald-700"
                          }`}>
                            {ds.task_type || "unknown"}
                          </span>
                        </div>
                        <div className="space-y-1 text-xs text-gray-500">
                          <p>Target: <strong className="text-gray-700">{ds.target_column || "Not set"}</strong></p>
                          {ds.version_metadata?.version && (
                            <p>Version: <strong className="text-gray-700">v{ds.version_metadata.version}</strong></p>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="bg-slate-950 text-slate-50 rounded-2xl p-5">
                    <p className="text-xs uppercase tracking-[0.22em] text-sky-300 mb-3">What You Can Do Here</p>
                    <div className="space-y-3 text-sm text-slate-200">
                      <p>Train multiple baselines side by side.</p>
                      <p>Use cross-validation for stronger evidence.</p>
                      <p>Enable tuning to search for better settings.</p>
                      <p>Compare feature importance and export a report.</p>
                    </div>
                  </div>
                  <div className="bg-white rounded-2xl border border-gray-200 p-5">
                    <h3 className="text-sm font-semibold text-gray-800 mb-3">Suggested Workflow</h3>
                    <ol className="space-y-2 text-sm text-gray-600">
                      <li>1. Pick a dataset with a target column already detected.</li>
                      <li>2. Start with all 3 model families.</li>
                      <li>3. Turn on 3-fold CV for a better comparison.</li>
                      <li>4. Add tuning if the baseline results are close.</li>
                    </ol>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="rounded-3xl border border-dashed border-gray-300 bg-gradient-to-br from-white via-slate-50 to-sky-50 p-10 text-center">
              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-sky-100 text-2xl">
                🧪
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">No datasets yet</h2>
              <p className="max-w-xl mx-auto text-sm leading-6 text-gray-600 mb-6">
                The experiments workspace comes alive once you upload data. After that, you can benchmark models, run cross-validation, tune settings, and export reports from one place.
              </p>
              <a
                href="/upload"
                className="inline-flex items-center rounded-xl bg-sky-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-sky-700"
              >
                Upload Your First Dataset
              </a>
            </div>
          )}
        </div>
      )}

      {selectedDataset && (
        <>
          {/* Training launcher */}
          <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Launch Training</h3>
            <div className="flex flex-wrap items-end gap-4">
              <div>
                <label className="text-xs text-gray-500 block mb-1">Models</label>
                <div className="flex gap-2">
                  {["linear", "xgboost", "random_forest"].map((m) => (
                    <label key={m} className="flex items-center gap-1.5 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={modelTypes.includes(m)}
                        onChange={(e) => {
                          if (e.target.checked) setModelTypes([...modelTypes, m]);
                          else setModelTypes(modelTypes.filter((t) => t !== m));
                        }}
                        className="rounded"
                      />
                      {m}
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-gray-500 block mb-1">Optimize For</label>
                <select
                  value={metric}
                  onChange={(e) => setMetric(e.target.value)}
                  className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm"
                >
                  {selectedDataset.task_type === "regression" ? (
                    <>
                      <option value="r2">R² Score</option>
                      <option value="rmse">RMSE (lower=better)</option>
                      <option value="mae">MAE (lower=better)</option>
                    </>
                  ) : (
                    <>
                      <option value="f1">F1 Score</option>
                      <option value="accuracy">Accuracy</option>
                      <option value="precision">Precision</option>
                      <option value="recall">Recall</option>
                    </>
                  )}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-500 block mb-1">Cross-Validation</label>
                <select
                  value={cvFolds}
                  onChange={(e) => setCvFolds(Number(e.target.value))}
                  className="border border-gray-200 rounded-lg px-3 py-1.5 text-sm"
                >
                  <option value={1}>Off</option>
                  <option value={3}>3 folds</option>
                  <option value={5}>5 folds</option>
                </select>
              </div>
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={tuneHyperparameters}
                  onChange={(e) => setTuneHyperparameters(e.target.checked)}
                  className="rounded"
                />
                Tune hyperparameters
              </label>
              <div>
                <p className="text-xs text-gray-500 mb-1">Target: <strong>{selectedDataset.target_column || "Not set"}</strong></p>
                <p className="text-xs text-gray-500">Task: <strong>{selectedDataset.task_type || "Not set"}</strong></p>
              </div>
              <button
                onClick={handleTrain}
                disabled={training || modelTypes.length === 0 || !selectedDataset.target_column}
                className="bg-blue-600 text-white text-sm font-medium px-5 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {training ? (
                  <span className="flex items-center gap-2">
                    <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                    Training...
                  </span>
                ) : (
                  "Train Models"
                )}
              </button>
            </div>
            {error && <p className="text-sm text-red-600 mt-3">{error}</p>}
          </div>

          {/* Training Results */}
          {trainResult && <TrainResults result={trainResult} taskType={selectedDataset.task_type || "classification"} />}

          {/* Past experiments */}
          {experiments.length > 0 && (
            <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Past Experiments</h3>
              <div className="space-y-2">
                {experiments.map((exp) => (
                  <button
                    key={exp.id}
                    onClick={() => loadExperiment(exp.id)}
                    className={`w-full text-left flex items-center justify-between p-3 rounded-lg border transition-colors ${
                      selectedExp?.experiment?.id === exp.id
                        ? "border-blue-300 bg-blue-50"
                        : "border-gray-100 hover:border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    <div>
                      <p className="text-sm font-medium text-gray-900">{exp.name || "Experiment"}</p>
                      <p className="text-xs text-gray-500">
                        {exp.status} · Optimized for {exp.optimization_metric}
                      </p>
                      {exp.tags && exp.tags.length > 0 && (
                        <p className="text-xs text-gray-400 mt-1">{exp.tags.join(", ")}</p>
                      )}
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      exp.archived
                        ? "bg-amber-100 text-amber-700"
                        : exp.status === "completed"
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-gray-100 text-gray-600"
                    }`}>
                      {exp.archived ? "archived" : exp.status}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Selected experiment detail */}
          {selectedExp && (
            <ExperimentDetail
              data={selectedExp}
              taskType={selectedDataset.task_type || "classification"}
              onDownloadReport={downloadReport}
            />
          )}
        </>
      )}
    </div>
  );
}

function TrainResults({ result, taskType }: { result: TrainResult; taskType: string }) {
  const jobs = result.jobs;
  const isClassification = taskType === "classification";
  const metricKeys = isClassification
    ? ["accuracy", "f1", "precision", "recall"]
    : ["r2", "rmse", "mae"];

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">Training Results</h3>

      {/* Feature engineering summary */}
      <div className="bg-gray-50 rounded-lg p-3 mb-4">
        <p className="text-xs font-medium text-gray-600 mb-1">Feature Engineering Applied:</p>
        <div className="space-y-0.5">
          {result.feature_engineering.transformations.map((t, i) => (
            <p key={i} className="text-xs text-gray-500">• {t}</p>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-2">
          {result.feature_engineering.feature_count} features · {result.feature_engineering.train_size} train / {result.feature_engineering.test_size} test samples
        </p>
        <p className="text-xs text-gray-500 mt-1">
          CV folds: {result.cv_folds || 1} · Hyperparameter tuning: {result.tune_hyperparameters ? "On" : "Off"}
        </p>
      </div>

      {/* Comparison table */}
      {jobs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-2 px-3 text-gray-600 font-medium">Model</th>
                {metricKeys.map((k) => (
                  <th key={k} className="text-right py-2 px-3 text-gray-600 font-medium">{k.toUpperCase()}</th>
                ))}
                <th className="text-right py-2 px-3 text-gray-600 font-medium">Time</th>
                <th className="text-center py-2 px-3 text-gray-600 font-medium">Best</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((r) => {
                const isBest = result.best_job_id === r.job_id;
                return (
                  <tr
                    key={r.job_id}
                    className={`border-b border-gray-100 ${isBest ? "bg-emerald-50" : ""}`}
                  >
                    <td className="py-2 px-3 font-medium text-gray-900">{r.model_type}</td>
                    {metricKeys.map((k) => (
                    <td key={k} className="text-right py-2 px-3 font-mono text-gray-700">
                        {r.metrics?.[k]?.toFixed(4) ?? "—"}
                      </td>
                    ))}
                    <td className="text-right py-2 px-3 text-gray-500">
                      {r.training_duration_seconds?.toFixed(1)}s
                    </td>
                    <td className="text-center py-2 px-3">
                      {isBest && <span className="text-emerald-600 font-bold">🏆</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Bar chart visualization */}
      {jobs.length > 0 && (
        <div className="mt-6">
          <h4 className="text-xs font-semibold text-gray-600 mb-3">Visual Comparison</h4>
          <div className="space-y-3">
            {metricKeys.filter(k => k !== "confusion_matrix").map((metricKey) => {
              const values = jobs.map((r) => r.metrics?.[metricKey] ?? 0);
              const maxVal = Math.max(...values.map(Math.abs), 0.001);
              const lowerIsBetter = ["rmse", "mae"].includes(metricKey);

              return (
                <div key={metricKey}>
                  <p className="text-xs text-gray-500 mb-1">{metricKey.toUpperCase()}{lowerIsBetter ? " (lower = better)" : ""}</p>
                  <div className="space-y-1">
                    {jobs.map((r) => {
                      const val = r.metrics?.[metricKey] ?? 0;
                      const pct = Math.abs(val) / maxVal * 100;
                      const isBest = result.best_job_id === r.job_id;
                      return (
                        <div key={r.job_id} className="flex items-center gap-2">
                          <span className="text-xs text-gray-600 w-28 truncate">{r.model_type}</span>
                          <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${
                                isBest ? "bg-emerald-500" : "bg-blue-400"
                              }`}
                              style={{ width: `${Math.min(pct, 100)}%` }}
                            />
                          </div>
                          <span className="text-xs font-mono text-gray-700 w-16 text-right">
                            {val.toFixed(4)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {jobs.some((job) => job.cross_validation) && (
        <div className="mt-6">
          <h4 className="text-xs font-semibold text-gray-600 mb-3">Cross-Validation Summary</h4>
          <div className="space-y-2">
            {jobs.map((job) => (
              <div key={job.job_id} className="rounded-lg border border-gray-100 px-3 py-2 text-xs text-gray-600">
                <strong className="text-gray-800">{job.model_type}</strong>:{" "}
                {job.cross_validation
                  ? Object.entries(job.cross_validation).map(([key, value]) => `${key}=${value}`).join(", ")
                  : "No CV run"}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ExperimentDetail({
  data,
  taskType,
  onDownloadReport,
}: {
  data: any;
  taskType: string;
  onDownloadReport: (experimentId: string) => void;
}) {
  const jobs: TrainingJob[] = data.jobs || [];
  const completed = jobs;
  const isClassification = taskType === "classification";
  const metricKeys = isClassification
    ? ["accuracy", "f1", "precision", "recall"]
    : ["r2", "rmse", "mae"];
  const [draftName, setDraftName] = useState(data.name || "");
  const [draftTags, setDraftTags] = useState((data.tags || []).join(", "));
  const [favorite, setFavorite] = useState(Boolean(data.favorite));
  const [archived, setArchived] = useState(Boolean(data.archived));

  async function saveMetadata() {
    await api.updateExperimentMetadata(data.experiment_id, {
      name: draftName,
      tags: draftTags.split(",").map((tag: string) => tag.trim()).filter(Boolean),
      favorite,
      archived,
    }).catch(() => {});
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">
        {data.name || "Experiment Detail"}
      </h3>
      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={() => onDownloadReport(data.experiment_id)}
          className="text-xs font-medium text-blue-700 hover:text-blue-900"
        >
          Export Markdown Report
        </button>
        {data.tags && data.tags.length > 0 && (
          <span className="text-xs text-gray-500">Tags: {data.tags.join(", ")}</span>
        )}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <input
          value={draftName}
          onChange={(e) => setDraftName(e.target.value)}
          placeholder="Experiment name"
          className="border border-gray-200 rounded-lg px-3 py-2 text-sm"
        />
        <input
          value={draftTags}
          onChange={(e) => setDraftTags(e.target.value)}
          placeholder="Tags, comma separated"
          className="border border-gray-200 rounded-lg px-3 py-2 text-sm"
        />
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <input type="checkbox" checked={favorite} onChange={(e) => setFavorite(e.target.checked)} className="rounded" />
          Favorite
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <input type="checkbox" checked={archived} onChange={(e) => setArchived(e.target.checked)} className="rounded" />
          Archived
        </label>
      </div>
      <button
        onClick={saveMetadata}
        className="text-xs font-medium text-blue-700 hover:text-blue-900 mb-4"
      >
        Save Experiment Metadata
      </button>

      {completed.length > 0 && (
        <>
          <div className="overflow-x-auto mb-6">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 text-gray-600 font-medium">Model</th>
                  {metricKeys.map((k) => (
                    <th key={k} className="text-right py-2 px-3 text-gray-600 font-medium">{k.toUpperCase()}</th>
                  ))}
                  <th className="text-right py-2 px-3 text-gray-600 font-medium">Duration</th>
                </tr>
              </thead>
              <tbody>
                {completed.map((j) => (
                  <tr key={j.id} className={`border-b border-gray-100 ${j.is_best ? "bg-emerald-50" : ""}`}>
                    <td className="py-2 px-3 font-medium text-gray-900 flex items-center gap-1.5">
                      {j.model_type}
                      {j.is_best && <span className="text-emerald-600">🏆</span>}
                    </td>
                    {metricKeys.map((k) => (
                      <td key={k} className="text-right py-2 px-3 font-mono text-gray-700">
                        {j.metrics?.[k]?.toFixed(4) ?? "—"}
                      </td>
                    ))}
                    <td className="text-right py-2 px-3 text-gray-500">
                      {j.training_duration_seconds?.toFixed(1)}s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Feature importance for best model */}
          {completed.filter((j) => j.is_best && j.feature_importance).map((j) => (
            <div key={j.id}>
              <h4 className="text-xs font-semibold text-gray-600 mb-2">
                Feature Importance ({j.model_type})
              </h4>
              <div className="space-y-1">
                {Object.entries(j.feature_importance || {})
                  .sort(([, a], [, b]) => (b as number) - (a as number))
                  .slice(0, 10)
                  .map(([feat, imp]) => {
                    const maxImp = Math.max(
                      ...Object.values(j.feature_importance || {}).map(Number)
                    );
                    const pct = maxImp > 0 ? ((imp as number) / maxImp) * 100 : 0;
                    return (
                      <div key={feat} className="flex items-center gap-2">
                        <span className="text-xs text-gray-600 w-36 truncate">{feat}</span>
                        <div className="flex-1 h-4 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-indigo-400 rounded-full"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono text-gray-500 w-14 text-right">
                          {(imp as number).toFixed(4)}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
