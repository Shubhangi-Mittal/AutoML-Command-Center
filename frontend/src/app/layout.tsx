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
      <body className="flex min-h-screen">
        {/* Sidebar */}
        <aside
          className={`${
            collapsed ? "w-16" : "w-56"
          } bg-slate-900 text-slate-300 flex flex-col transition-all duration-200 shrink-0`}
        >
          <div className="p-4 border-b border-slate-700 flex items-center justify-between">
            {!collapsed && (
              <div>
                <h1 className="text-white font-bold text-sm">AutoML</h1>
                <p className="text-xs text-slate-500">Command Center</p>
              </div>
            )}
            <button
              onClick={() => setCollapsed(!collapsed)}
              className="text-slate-500 hover:text-white text-lg"
            >
              {collapsed ? "→" : "←"}
            </button>
          </div>

          <nav className="flex-1 py-4">
            {NAV_ITEMS.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                    active
                      ? "bg-blue-600/20 text-blue-400 border-r-2 border-blue-400"
                      : "hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  <span className="text-lg">{item.icon}</span>
                  {!collapsed && <span>{item.label}</span>}
                </Link>
              );
            })}
          </nav>

          {!collapsed && (
            <div className="p-4 border-t border-slate-700 text-xs text-slate-500">
              Divyesh Mistry
              <br />v1.0.0
            </div>
          )}
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-auto">{children}</main>
      </body>
    </html>
  );
}
