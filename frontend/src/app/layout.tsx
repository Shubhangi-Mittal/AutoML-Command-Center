"use client";
import "./globals.css";
import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: "📊" },
  { href: "/upload", label: "Upload", icon: "📁" },
  { href: "/agent", label: "AI Agent", icon: "🤖" },
  { href: "/experiments", label: "Experiments", icon: "🧪" },
  { href: "/deploy", label: "Deploy", icon: "🚀" },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <html lang="en">
      <body className="app-shell">
        {/* Sidebar */}
        <aside
          className={`${
            collapsed ? "w-20" : "w-72"
          } app-sidebar transition-all duration-300 shrink-0`}
        >
          <div className="p-5 border-b border-white/10 flex items-center justify-between">
            {!collapsed && (
              <div className="space-y-2">
                <div className="inline-flex items-center rounded-full border border-sky-400/25 bg-sky-400/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-200">
                  Student Build
                </div>
                <div>
                  <h1 className="font-display text-white font-bold text-lg leading-none">AutoML</h1>
                  <p className="text-xs text-slate-400 mt-1">Command Center</p>
                </div>
              </div>
            )}
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="flex h-10 w-10 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-slate-400 transition hover:border-sky-300/30 hover:text-white"
            >
              {collapsed ? "→" : "←"}
            </button>
          </div>

          <div className="px-4 pt-5">
            {!collapsed && (
              <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-4 text-sm text-slate-300 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]">
                <p className="font-semibold text-white">Build, compare, deploy.</p>
                <p className="mt-1 text-xs leading-5 text-slate-400">
                  A compact ML workspace with dataset-aware AI assistance.
                </p>
              </div>
            )}
          </div>

          <nav className="flex-1 py-5 px-3">
            {NAV_ITEMS.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`mb-2 flex items-center gap-3 rounded-2xl px-4 py-3 text-sm transition-all ${
                    active
                      ? "bg-[linear-gradient(135deg,rgba(14,165,233,0.22),rgba(59,130,246,0.10))] text-white shadow-[0_16px_40px_rgba(14,165,233,0.18)]"
                      : "text-slate-300 hover:bg-white/[0.06] hover:text-white"
                  }`}
                >
                  <span className={`text-lg ${active ? "" : "opacity-80"}`}>{item.icon}</span>
                  {!collapsed && (
                    <div className="flex-1">
                      <span className="block font-medium">{item.label}</span>
                      {active && <span className="block text-[11px] text-sky-100/70">Active workspace</span>}
                    </div>
                  )}
                </Link>
              );
            })}
          </nav>

          {!collapsed && (
            <div className="p-5 border-t border-white/10 text-xs text-slate-500">
              <p>AutoML Command Center</p>
              <p className="mt-1">v1.0.0</p>
            </div>
          )}
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-auto">
          <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.10),_transparent_22%),radial-gradient(circle_at_top_right,_rgba(16,185,129,0.10),_transparent_18%),linear-gradient(180deg,_#f8fbff_0%,_#f8fafc_36%,_#f3f7fb_100%)]">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
