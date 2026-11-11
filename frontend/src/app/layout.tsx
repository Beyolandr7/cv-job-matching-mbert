import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sistem Pencocokan CV & Lowongan Kerja (NER + Siamese mBERT + Cosine)",
  description:
    "Implementasi Deep Learning-Based NER & Siamese Multilingual BERT (mBERT) dengan Cosine Similarity untuk Rekrutmen",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}

