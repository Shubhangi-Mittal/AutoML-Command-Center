"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Dataset, Experiment, ServingStatus } from "@/types";

export default function DashboardPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [serving, setServing] = useState<ServingStatus | null>(null);
  const [backendStatus, setBackendStatus] = useState<"loading" | "ok" | "error">("loading");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboard();
  }, []);

  async function loadDashboard() {
    try {
      const health = await api.health();
      setBackendStatus(health.status === "ok" ? "ok" : "error");

      const [ds, exps, status] = await Promise.allSettled([
        api.listDatasets(),
        api.listExperiments(),
        api.servingStatus(),
      ]);

      if (ds.status === "fulfilled") setDatasets(ds.value);
      if (exps.status === "fulfilled") setExperiments(exps.value);
      if (status.status === "fulfilled") setServing(status.value);
    } catch {
      setBackendStatus("error");
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mx-auto" />
          <p className="mt-4 text-gray-500 text-sm">Connecting to backend...</p>
          <p className="mt-1 text-gray-400 text-xs">May take ~30s on free tier</p>
        </div>
      </div>
    );
  }

  const completedExps = experiments.filter((e) => e.status === "completed");
  const totalJobs = experiments.length;

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 text-sm mt-1">
          AutoML Command Center — your AI-powered data science platform
        </p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <StatusCard
          icon="🔗"
          label="Backend"
          value={backendStatus === "ok" ? "Connected" : "Offline"}
          color={backendStatus === "ok" ? "green" : "red"}
        />
        <StatusCard
          icon="📁"
          label="Datasets"
          value={String(datasets.length)}
          color="blue"
        />
        <StatusCard
          icon="🧪"
          label="Experiments"
          value={`${completedExps.length} / ${totalJobs}`}
          color="purple"
        />
        <StatusCard
          icon="🚀"
          label="Model Serving"
          value={serving?.status === "active" ? "Active" : "No Model"}
          color={serving?.status === "active" ? "green" : "gray"}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Link
          href="/upload"
          className="bg-white rounded-lg border border-gray-200 p-5 hover:border-blue-300 hover:shadow-md transition-all group"
        >
          <div className="text-2xl mb-2">📁</div>
          <h3 className="font-semibold text-gray-900 group-hover:text-blue-600">
            Upload Dataset
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            Upload a CSV and get instant profiling
          </p>
        </Link>
        <Link
          href="/agent"
          className="bg-white rounded-lg border border-gray-200 p-5 hover:border-blue-300 hover:shadow-md transition-all group"
        >
          <div className="text-2xl mb-2">🤖</div>
          <h3 className="font-semibold text-gray-900 group-hover:text-blue-600">
            AI Agent
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            Chat with your ML co-pilot
          </p>
        </Link>
        <Link
          href="/experiments"
          className="bg-white rounded-lg border border-gray-200 p-5 hover:border-blue-300 hover:shadow-md transition-all group"
        >
          <div className="text-2xl mb-2">🧪</div>
          <h3 className="font-semibold text-gray-900 group-hover:text-blue-600">
            Experiments
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            Compare models and view results
          </p>
        </Link>
      </div>

      {/* Recent Datasets */}
      {datasets.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
          <h2 className="font-semibold text-gray-900 mb-3">Recent Datasets</h2>
          <div className="space-y-2">
            {datasets.slice(0, 5).map((ds) => (
              <Link
                key={ds.id}
                href={`/upload?id=${ds.id}`}
                className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg">📄</span>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{ds.name}</p>
                    <p className="text-xs text-gray-500">
                      {ds.rows?.toLocaleString()} rows × {ds.columns} cols
                      {ds.target_column && ` · Target: ${ds.target_column}`}
                    </p>
                  </div>
                </div>
                <span className="text-xs text-gray-400">
                  {ds.task_type || "—"}
                </span>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {datasets.length === 0 && (
        <div className="bg-white rounded-lg border border-dashed border-gray-300 p-12 text-center">
          <div className="text-4xl mb-3">📊</div>
          <h3 className="font-semibold text-gray-900 mb-1">No datasets yet</h3>
          <p className="text-sm text-gray-500 mb-4">
            Upload a CSV to get started with automated profiling and model training
          </p>
          <Link
            href="/upload"
            className="inline-block bg-blue-600 text-white text-sm font-medium px-4 py-2 rounded-lg hover:bg-blue-700"
          >
            Upload Your First Dataset
          </Link>
        </div>
      )}
    </div>
  );
}

function StatusCard({
  icon,
  label,
  value,
  color,
}: {
  icon: string;
  label: string;
  value: string;
  color: string;
}) {
  const colorMap: Record<string, string> = {
    green: "bg-emerald-50 text-emerald-700 border-emerald-200",
    blue: "bg-blue-50 text-blue-700 border-blue-200",
    purple: "bg-purple-50 text-purple-700 border-purple-200",
    red: "bg-red-50 text-red-700 border-red-200",
    gray: "bg-gray-50 text-gray-600 border-gray-200",
  };

  return (
    <div className={`rounded-lg border p-4 ${colorMap[color] || colorMap.gray}`}>
      <div className="flex items-center gap-2 mb-1">
        <span>{icon}</span>
        <span className="text-xs font-medium opacity-75">{label}</span>
      </div>
      <p className="text-lg font-bold">{value}</p>
    </div>
  );
}
