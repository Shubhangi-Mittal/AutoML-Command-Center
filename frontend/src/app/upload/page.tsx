"use client";
import { useState, useCallback, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import { Dataset, ColumnProfile, DatasetVersionMetadata } from "@/types";

export default function UploadPage() {
  return (
    <Suspense fallback={<div className="p-6 text-gray-500">Loading...</div>}>
      <UploadPageContent />
    </Suspense>
  );
}

function UploadPageContent() {
  const searchParams = useSearchParams();
  const preselectedId = searchParams.get("id");

  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [versions, setVersions] = useState<DatasetVersionMetadata[]>([]);

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch(() => {});
    if (preselectedId) {
      api.getDataset(preselectedId).then(setDataset).catch(() => {});
    }
  }, [preselectedId]);

  useEffect(() => {
    if (!dataset?.id) {
      setVersions([]);
      return;
    }
    api.getDatasetVersions(dataset.id)
      .then((result) => setVersions(result.versions || []))
      .catch(() => setVersions([]));
  }, [dataset?.id]);

  const handleUpload = useCallback(async (file: File) => {
    if (!file.name.endsWith(".csv")) {
      setError("Only CSV files are supported");
      return;
    }
    setUploading(true);
    setError("");
    try {
      const result = await api.uploadDataset(file);
      setDataset(result);
      api.listDatasets().then(setDatasets).catch(() => {});
    } catch (e: any) {
      setError(e.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) handleUpload(file);
    },
    [handleUpload]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-1">Upload Dataset</h1>
      <p className="text-gray-500 text-sm mb-6">
        Upload a CSV file to get instant automated profiling
      </p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-10 text-center transition-all mb-6 ${
          dragging
            ? "border-blue-400 bg-blue-50"
            : "border-gray-300 hover:border-gray-400 bg-white"
        }`}
      >
        {uploading ? (
          <div>
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto" />
            <p className="mt-3 text-sm text-gray-600">Uploading and profiling...</p>
          </div>
        ) : (
          <>
            <div className="text-4xl mb-3">📄</div>
            <p className="text-sm text-gray-700 font-medium">
              Drag and drop a CSV file here
            </p>
            <p className="text-xs text-gray-400 mt-1 mb-3">or</p>
            <label className="inline-block bg-blue-600 text-white text-sm font-medium px-4 py-2 rounded-lg hover:bg-blue-700 cursor-pointer">
              Browse Files
              <input
                type="file"
                accept=".csv"
                onChange={handleFileInput}
                className="hidden"
              />
            </label>
          </>
        )}
      </div>

      {error && (
        <div className="bg-red-50 text-red-700 text-sm p-3 rounded-lg mb-6 border border-red-200">
          {error}
        </div>
      )}

      {/* Existing datasets */}
      {!dataset && datasets.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">
            Or select an existing dataset:
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {datasets.map((ds) => (
              <button
                key={ds.id}
                onClick={() => api.getDataset(ds.id).then(setDataset).catch(() => setDataset(ds))}
                className="flex items-center gap-3 p-3 rounded-lg border border-gray-100 hover:border-blue-300 hover:bg-blue-50 transition-all text-left"
              >
                <span>📄</span>
                <div>
                  <p className="text-sm font-medium text-gray-900">{ds.name}</p>
                  <p className="text-xs text-gray-500">
                    {ds.rows?.toLocaleString()} rows × {ds.columns} cols
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Profile viewer */}
      {dataset?.profile && <ProfileViewer dataset={dataset} versions={versions} />}
    </div>
  );
}

function ProfileViewer({ dataset, versions }: { dataset: Dataset; versions: DatasetVersionMetadata[] }) {
  const profile = dataset.profile!;
  const columns = profile.columns || {};
  const warnings = profile.warnings || [];
  const correlations = profile.correlations?.top_pairs || [];

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="bg-white rounded-lg border border-gray-200 p-5">
        <h2 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <span>📊</span> Dataset Profile: {dataset.name}
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <Stat label="Rows" value={profile.row_count.toLocaleString()} />
          <Stat label="Columns" value={String(profile.column_count)} />
          <Stat label="Memory" value={`${profile.memory_usage_mb} MB`} />
          <Stat label="Duplicates" value={String(profile.duplicate_rows)} />
          <Stat
            label="Target"
            value={profile.suggested_target || "—"}
            highlight
          />
        </div>
        {profile.suggested_task_type && (
          <div className="mt-3 inline-block bg-blue-50 text-blue-700 text-xs font-medium px-3 py-1 rounded-full">
            Suggested task: {profile.suggested_task_type}
          </div>
        )}
        {dataset.version_metadata?.version && (
          <div className="mt-2 text-xs text-gray-500">
            Dataset version: <strong>v{dataset.version_metadata.version}</strong>
          </div>
        )}
      </div>

      {versions.length > 1 && (
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-900 mb-3">Dataset Version History</h3>
          <div className="space-y-2">
            {versions.map((version) => (
              <div
                key={version.dataset_id}
                className={`rounded-lg border px-3 py-2 text-sm ${
                  version.dataset_id === dataset.id ? "border-blue-300 bg-blue-50" : "border-gray-200"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-gray-800">v{version.version}</span>
                  <span className="text-xs text-gray-500">{version.dataset_id.slice(0, 8)}...</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-amber-50 rounded-lg border border-amber-200 p-4">
          <h3 className="font-semibold text-amber-800 text-sm mb-2">
            ⚠️ Data Quality Warnings
          </h3>
          <ul className="space-y-1">
            {warnings.map((w, i) => (
              <li key={i} className="text-sm text-amber-700">
                • {w}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Column cards */}
      <div>
        <h3 className="font-semibold text-gray-900 mb-3">Column Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.entries(columns).map(([name, info]) => (
            <ColumnCard
              key={name}
              name={name}
              info={info}
              isTarget={name === profile.suggested_target}
            />
          ))}
        </div>
      </div>

      {/* Top correlations */}
      {correlations.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <h3 className="font-semibold text-gray-900 mb-3">
            Top Correlations
          </h3>
          <div className="space-y-2">
            {correlations.slice(0, 8).map((pair, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-gray-500 w-6">{i + 1}.</span>
                <span className="text-sm font-medium text-gray-700 w-32 truncate">
                  {pair.col1}
                </span>
                <span className="text-xs text-gray-400">↔</span>
                <span className="text-sm font-medium text-gray-700 w-32 truncate">
                  {pair.col2}
                </span>
                <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      Math.abs(pair.correlation) > 0.7
                        ? "bg-red-400"
                        : Math.abs(pair.correlation) > 0.4
                        ? "bg-amber-400"
                        : "bg-blue-400"
                    }`}
                    style={{ width: `${Math.abs(pair.correlation) * 100}%` }}
                  />
                </div>
                <span className="text-sm font-mono text-gray-600 w-14 text-right">
                  {pair.correlation.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ColumnCard({
  name,
  info,
  isTarget,
}: {
  name: string;
  info: ColumnProfile;
  isTarget: boolean;
}) {
  const isNumeric = info.dtype_category === "numeric";

  return (
    <div
      className={`bg-white rounded-lg border p-4 ${
        isTarget ? "border-blue-300 ring-1 ring-blue-100" : "border-gray-200"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-semibold text-gray-900 truncate">{name}</h4>
        <div className="flex items-center gap-1.5">
          {isTarget && (
            <span className="bg-blue-100 text-blue-700 text-[10px] font-medium px-1.5 py-0.5 rounded">
              TARGET
            </span>
          )}
          <span
            className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${
              isNumeric
                ? "bg-emerald-100 text-emerald-700"
                : "bg-purple-100 text-purple-700"
            }`}
          >
            {info.dtype_category}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <div className="text-gray-500">Missing</div>
        <div
          className={`text-right font-medium ${
            info.missing_pct > 10 ? "text-red-600" : "text-gray-700"
          }`}
        >
          {info.missing_pct}%
        </div>
        <div className="text-gray-500">Unique</div>
        <div className="text-right font-medium text-gray-700">
          {info.unique_count.toLocaleString()}
        </div>

        {isNumeric && info.mean != null && (
          <>
            <div className="text-gray-500">Mean</div>
            <div className="text-right font-medium text-gray-700">
              {info.mean.toFixed(2)}
            </div>
            <div className="text-gray-500">Std</div>
            <div className="text-right font-medium text-gray-700">
              {info.std?.toFixed(2)}
            </div>
            <div className="text-gray-500">Range</div>
            <div className="text-right font-medium text-gray-700">
              {info.min?.toFixed(1)} — {info.max?.toFixed(1)}
            </div>
          </>
        )}

        {!isNumeric && info.top_values && (
          <>
            <div className="col-span-2 mt-1 text-gray-500">Top values:</div>
            {Object.entries(info.top_values)
              .slice(0, 3)
              .map(([val, count]) => (
                <div key={val} className="col-span-2 flex justify-between">
                  <span className="text-gray-600 truncate max-w-[60%]">{val}</span>
                  <span className="text-gray-500">{count}</span>
                </div>
              ))}
          </>
        )}
      </div>

      {/* Mini histogram for numeric columns */}
      {isNumeric && info.histogram && info.histogram.length > 0 && (
        <div className="mt-2 flex items-end gap-px h-8">
          {info.histogram.map((bin, i) => {
            const maxCount = Math.max(...info.histogram!.map((b) => b.count));
            const height = maxCount > 0 ? (bin.count / maxCount) * 100 : 0;
            return (
              <div
                key={i}
                className="flex-1 bg-blue-300 rounded-t-sm min-h-[2px]"
                style={{ height: `${height}%` }}
                title={`${bin.bin_start.toFixed(1)}–${bin.bin_end.toFixed(1)}: ${bin.count}`}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <p className="text-xs text-gray-500">{label}</p>
      <p
        className={`text-lg font-bold ${
          highlight ? "text-blue-600" : "text-gray-900"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
