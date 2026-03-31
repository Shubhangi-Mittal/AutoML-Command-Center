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
      const health = await waitForBackend();
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
      <div className="flex items-center justify-center min-h-screen p-6">
        <div className="glass-card rounded-[2rem] p-10 text-center max-w-md w-full">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-3xl bg-sky-100 text-3xl">⚙️</div>
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-sky-600 mx-auto mt-5" />
          <p className="mt-4 text-slate-600 text-sm">Connecting to backend...</p>
          <p className="mt-1 text-slate-400 text-xs">May take ~30s on free tier</p>
        </div>
      </div>
    );
  }

  const completedExps = experiments.filter((e) => e.status === "completed");
  const totalJobs = experiments.length;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="hero-grid relative overflow-hidden rounded-[2rem] border border-slate-200/80 bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.16),_transparent_24%),radial-gradient(circle_at_top_right,_rgba(16,185,129,0.12),_transparent_20%),linear-gradient(135deg,_rgba(255,255,255,0.94),_rgba(247,250,252,0.90))] px-8 py-10 mb-8 shadow-[0_24px_70px_rgba(15,23,42,0.07)]">
        <div className="max-w-3xl">
          <p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-600 mb-3">Control Room</p>
          <h1 className="font-display text-4xl md:text-5xl font-bold text-slate-900 leading-tight">
            Build the whole ML workflow from one polished workspace.
          </h1>
          <p className="text-slate-600 text-sm md:text-base mt-4 max-w-2xl leading-7">
            Upload data, explore quality, run model experiments, deploy a winner, and steer the workflow with an AI copilot that remembers context per dataset.
          </p>
          <div className="flex flex-wrap gap-3 mt-6">
            <Link href="/upload" className="rounded-2xl bg-sky-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-700">
              Upload Dataset
            </Link>
            <Link href="/experiments" className="rounded-2xl border border-slate-300 bg-white/80 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-sky-300 hover:text-sky-700">
              Open Experiments
            </Link>
          </div>
        </div>
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
          value={serving?.status === "deployed" ? "Active" : "No Model"}
          color={serving?.status === "deployed" ? "green" : "gray"}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Link
          href="/upload"
          className="glass-card rounded-[1.75rem] p-6 hover:border-sky-300 hover:-translate-y-1 transition-all group"
        >
          <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-sky-100 text-2xl">📁</div>
          <h3 className="font-display font-semibold text-gray-900 group-hover:text-sky-600">
            Upload Dataset
          </h3>
          <p className="text-sm text-gray-500 mt-2 leading-6">
            Upload a CSV and get instant profiling
          </p>
        </Link>
        <Link
          href="/agent"
          className="glass-card rounded-[1.75rem] p-6 hover:border-sky-300 hover:-translate-y-1 transition-all group"
        >
          <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-100 text-2xl">🤖</div>
          <h3 className="font-display font-semibold text-gray-900 group-hover:text-sky-600">
            AI Agent
          </h3>
          <p className="text-sm text-gray-500 mt-2 leading-6">
            Chat with your ML co-pilot
          </p>
        </Link>
        <Link
          href="/experiments"
          className="glass-card rounded-[1.75rem] p-6 hover:border-sky-300 hover:-translate-y-1 transition-all group"
        >
          <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-100 text-2xl">🧪</div>
          <h3 className="font-display font-semibold text-gray-900 group-hover:text-sky-600">
            Experiments
          </h3>
          <p className="text-sm text-gray-500 mt-2 leading-6">
            Compare models and view results
          </p>
        </Link>
      </div>

      {/* Recent Datasets */}
      {datasets.length > 0 && (
        <div className="glass-card rounded-[1.75rem] p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-display font-semibold text-gray-900 text-xl">Recent Datasets</h2>
            <span className="text-xs uppercase tracking-[0.22em] text-slate-400">Dataset Library</span>
          </div>
          <div className="space-y-2">
            {datasets.slice(0, 5).map((ds) => (
              <Link
                key={ds.id}
                href={`/upload?id=${ds.id}`}
                className="flex items-center justify-between p-4 rounded-2xl border border-transparent hover:bg-white hover:border-sky-200 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-100 text-lg">📄</span>
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
        <div className="glass-card rounded-[2rem] border-dashed p-12 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-3xl bg-sky-100 text-3xl">📊</div>
          <h3 className="font-display text-2xl font-semibold text-gray-900 mb-2">No datasets yet</h3>
          <p className="text-sm text-gray-500 mb-5 max-w-lg mx-auto leading-6">
            Upload a CSV to get started with automated profiling and model training
          </p>
          <Link
            href="/upload"
            className="inline-block rounded-2xl bg-sky-600 text-white text-sm font-medium px-5 py-3 hover:bg-sky-700"
          >
            Upload Your First Dataset
          </Link>
        </div>
      )}
    </div>
  );
}

async function waitForBackend(maxAttempts = 6, delayMs = 3500) {
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await api.health();
    } catch (error) {
      lastError = error;
      if (attempt < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }
    }
  }

  throw lastError ?? new Error("Backend did not respond");
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
    <div className={`glass-card rounded-[1.5rem] p-5 ${colorMap[color] || colorMap.gray}`}>
      <div className="flex items-center gap-2 mb-1">
        <span>{icon}</span>
        <span className="text-xs font-medium opacity-75">{label}</span>
      </div>
      <p className="font-display text-2xl font-bold">{value}</p>
    </div>
  );
}
