"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { Dataset, TrainResult, TrainingJob } from "@/types";

export default function ExperimentsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [selectedExp, setSelectedExp] = useState<any>(null);
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);
  const [modelTypes, setModelTypes] = useState(["linear", "xgboost", "random_forest"]);
  const [metric, setMetric] = useState("f1");
  const [error, setError] = useState("");

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch(() => {});
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      api.listExperiments(selectedDataset.id).then(setExperiments).catch(() => {});
    }
  }, [selectedDataset]);

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
      });
      setTrainResult(result);
      api.listExperiments(selectedDataset.id).then(setExperiments).catch(() => {});
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTraining(false);
    }
  }

  async function loadExperiment(expId: string) {
    try {
      const data = await api.getExperiment(expId);
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
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      exp.status === "completed" ? "bg-emerald-100 text-emerald-700" : "bg-gray-100 text-gray-600"
                    }`}>
                      {exp.status}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Selected experiment detail */}
          {selectedExp && <ExperimentDetail data={selectedExp} taskType={selectedDataset.task_type || "classification"} />}
        </>
      )}
    </div>
  );
}

function TrainResults({ result, taskType }: { result: TrainResult; taskType: string }) {
  const completed = result.results.filter((r) => r.status === "completed");
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
          {result.feature_engineering.n_features} features · {result.feature_engineering.n_train} train / {result.feature_engineering.n_test} test samples
        </p>
      </div>

      {/* Comparison table */}
      {completed.length > 0 && (
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
              {completed.map((r) => {
                const isBest = result.best_model?.job_id === r.job_id;
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
                      {r.metrics?.training_duration?.toFixed(1)}s
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

      {/* Failed models */}
      {result.results.filter((r) => r.status === "failed").map((r) => (
        <div key={r.job_id} className="mt-2 text-xs text-red-600">
          {r.model_type} failed: {r.error}
        </div>
      ))}

      {/* Bar chart visualization */}
      {completed.length > 0 && (
        <div className="mt-6">
          <h4 className="text-xs font-semibold text-gray-600 mb-3">Visual Comparison</h4>
          <div className="space-y-3">
            {metricKeys.filter(k => k !== "confusion_matrix").map((metricKey) => {
              const values = completed.map((r) => r.metrics?.[metricKey] ?? 0);
              const maxVal = Math.max(...values.map(Math.abs), 0.001);
              const lowerIsBetter = ["rmse", "mae"].includes(metricKey);

              return (
                <div key={metricKey}>
                  <p className="text-xs text-gray-500 mb-1">{metricKey.toUpperCase()}{lowerIsBetter ? " (lower = better)" : ""}</p>
                  <div className="space-y-1">
                    {completed.map((r) => {
                      const val = r.metrics?.[metricKey] ?? 0;
                      const pct = Math.abs(val) / maxVal * 100;
                      const isBest = result.best_model?.job_id === r.job_id;
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
    </div>
  );
}

function ExperimentDetail({ data, taskType }: { data: any; taskType: string }) {
  const jobs: TrainingJob[] = data.jobs || [];
  const completed = jobs.filter((j) => j.status === "completed");
  const isClassification = taskType === "classification";
  const metricKeys = isClassification
    ? ["accuracy", "f1", "precision", "recall"]
    : ["r2", "rmse", "mae"];

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
      <h3 className="text-sm font-semibold text-gray-700 mb-4">
        {data.experiment?.name || "Experiment Detail"}
      </h3>

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
