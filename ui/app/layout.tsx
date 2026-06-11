import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PINN · Heat Equation",
  description: "Physics-Informed Neural Network — 1D pipe with moving gas burner",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0f1117] text-slate-100 antialiased">{children}</body>
    </html>
  );
}
