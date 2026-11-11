"use client";

import React, { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1";

interface PresetJob {
  id: string;
  domain: string;
  job_title: string;
  job_description: string;
}

interface ExtractedEntities {
  EDU?: string[];
  ORG?: string[];
  SKILL?: string[];
  ROLE?: string[];
  DUR?: string[];
  ner_tagged_preview?: string;
  experience_years_detected?: number | null;
  detected_domain?: string;
  job_detected_domain?: string;
  raw_cosine_similarity?: number;
  fine_tuned_model_loaded?: boolean;
}

interface AnalysisResult {
  id?: string;
  cv_filename: string;
  job_title: string;
  similarity_score: number;
  similarity_percentage: number;
  status_category: "Fit" | "Close Fit" | "Not Fit" | string;
  extracted_entities: ExtractedEntities;
  cv_text_preview: string;
  job_description_preview: string;
  cv_text_full?: string;
  job_text_full?: string;
  is_validated?: boolean;
  created_at?: string;
}

interface StatsResponse {
  history_summary: {
    total_processed: number;
    validated_count: number;
    counts: Record<string, number>;
    percentages: Record<string, number>;
  };
  dataset_summary: {
    dataset_filename: string;
    total_rows: number;
    initial_rows: number;
    newly_validated_rows: number;
    counts: Record<string, number>;
    percentages: Record<string, number>;
  };
  recent_history: Array<{
    id: string;
    cv_filename: string;
    job_title: string;
    similarity_score: number;
    similarity_percentage: number;
    status_category: string;
    is_validated: boolean;
    created_at: string;
  }>;
}

interface RetrainMetrics {
  epochs_trained: number;
  samples_trained: number;
  initial_loss: number;
  final_loss: number;
  loss_history: number[];
  before_similarity_percentage: number;
  after_similarity_percentage: number;
  total_dataset_rows: number;
  duration_seconds: number;
  model_checkpoint: string;
}

export default function HomePage() {
  // Authentication state (User Biasa vs Admin)
  const [token, setToken] = useState<string | null>(null);
  const [adminUsername, setAdminUsername] = useState<string | null>(null);
  const isAdmin = Boolean(token);

  // Native <dialog> ref for Admin Sign-In Modal
  const loginDialogRef = useRef<HTMLDialogElement | null>(null);
  const [usernameInput, setUsernameInput] = useState("admin");
  const [passwordInput, setPasswordInput] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loginLoading, setLoginLoading] = useState(false);

  // Preset jobs & form inputs
  const [presetJobs, setPresetJobs] = useState<PresetJob[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>("banking-ops-manager");
  const [jobTitle, setJobTitle] = useState<string>("Consumer Banking Operations Manager");
  const [jobDescription, setJobDescription] = useState<string>(
    "Seeking an experienced Banking Operations Manager & Default Management Supervisor with a Bachelor of Science in Finance or Business Administration. Must have 10+ years of experience in customer service management, call center operations, cash deposits, account resolution, staffing, scheduling, budgeting, payroll, and process improvement."
  );
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);

  // Analysis output & Admin interactive validation controls (Combobox + Numeric Up-Down)
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [adminCategory, setAdminCategory] = useState<string>("Fit");
  const [adminPercentage, setAdminPercentage] = useState<number>(85.0);
  const [validating, setValidating] = useState(false);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [retrainMetrics, setRetrainMetrics] = useState<RetrainMetrics | null>(null);

  // Admin Dashboard Stats & Charts
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [retrainingStandalone, setRetrainingStandalone] = useState(false);

  // Restore saved token & fetch preset jobs on mount
  useEffect(() => {
    const savedToken = window.localStorage.getItem("mbert_admin_token");
    const savedUser = window.localStorage.getItem("mbert_admin_username");
    if (savedToken) {
      setToken(savedToken);
      setAdminUsername(savedUser || "admin");
    }

    fetch(`${API_BASE}/matching/preset-jobs`)
      .then((r) => r.json())
      .then((data) => {
        if (data?.items?.length) {
          setPresetJobs(data.items);
        }
      })
      .catch(() => {
        // Ignore if backend is still starting up
      });
  }, []);

  // Progressive enhancement fallback for <dialog closedby="any"> light-dismiss
  useEffect(() => {
    const dialog = loginDialogRef.current;
    if (!dialog || "closedBy" in HTMLDialogElement.prototype) return;

    let pressedOutside = false;
    const isOutside = (e: PointerEvent, el: HTMLDialogElement) => {
      const r = el.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    const onDown = (e: PointerEvent) => {
      pressedOutside = isOutside(e, dialog);
    };
    const onUp = (e: PointerEvent) => {
      if (pressedOutside && isOutside(e, dialog)) {
        dialog.close();
      }
    };
    dialog.addEventListener("pointerdown", onDown);
    dialog.addEventListener("pointerup", onUp);
    return () => {
      dialog.removeEventListener("pointerdown", onDown);
      dialog.removeEventListener("pointerup", onUp);
    };
  }, []);

  // Load Admin stats whenever token is active
  const fetchAdminStats = async (activeToken?: string | null) => {
    const tk = activeToken ?? token;
    if (!tk) return;
    try {
      const resp = await fetch(`${API_BASE}/matching/stats`, {
        headers: { Authorization: `Bearer ${tk}` },
      });
      if (resp.ok) {
        const data: StatsResponse = await resp.json();
        setStats(data);
      } else if (resp.status === 401) {
        handleLogout();
      }
    } catch {
      // Ignore network error during refresh
    }
  };

  useEffect(() => {
    if (token) {
      fetchAdminStats(token);
    } else {
      setStats(null);
    }
  }, [token]);

  const openLoginModal = () => {
    setLoginError(null);
    loginDialogRef.current?.showModal();
  };

  const closeLoginModal = () => {
    if (loginDialogRef.current) {
      loginDialogRef.current.returnValue = "";
      loginDialogRef.current.close();
    }
  };

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError(null);
    setLoginLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: usernameInput.trim(),
          password: passwordInput,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Gagal melakukan autentikasi Admin.");
      }
      window.localStorage.setItem("mbert_admin_token", data.access_token);
      window.localStorage.setItem("mbert_admin_username", data.user?.username || "admin");
      setToken(data.access_token);
      setAdminUsername(data.user?.username || "admin");
      setPasswordInput("");
      closeLoginModal();
    } catch (err: any) {
      setLoginError(err.message || "Terjadi kesalahan saat login.");
    } finally {
      setLoginLoading(false);
    }
  };

  const handleLogout = () => {
    window.localStorage.removeItem("mbert_admin_token");
    window.localStorage.removeItem("mbert_admin_username");
    setToken(null);
    setAdminUsername(null);
    setValidationMessage(null);
  };

  const handlePresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    setSelectedPresetId(val);
    const found = presetJobs.find((j) => j.id === val);
    if (found) {
      setJobTitle(found.job_title);
      setJobDescription(found.job_description);
    }
  };

  const handleAnalyzeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAnalyzeError(null);
    setValidationMessage(null);
    setRetrainMetrics(null);

    if (!selectedFile) {
      setAnalyzeError("Silakan pilih berkas CV berformat PDF terlebih dahulu.");
      return;
    }
    if (!jobTitle.trim() || !jobDescription.trim()) {
      setAnalyzeError("Judul Posisi dan Deskripsi Lowongan wajib diisi.");
      return;
    }

    setAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("job_title", jobTitle.trim());
      formData.append("job_description", jobDescription.trim());

      const headers: Record<string, string> = {};
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }

      const resp = await fetch(`${API_BASE}/matching/analyze`, {
        method: "POST",
        headers,
        body: formData,
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Gagal menganalisis berkas CV.");
      }

      setResult(data);
      setAdminCategory(data.status_category);
      setAdminPercentage(data.similarity_percentage);
      if (token) {
        fetchAdminStats(token);
      }
    } catch (err: any) {
      setAnalyzeError(err.message || "Gagal menghubungi server backend di port 8000.");
    } finally {
      setAnalyzing(false);
    }
  };

  // Sync Admin Combobox & Numeric Up-Down
  const handleAdminCategoryChange = (newCat: string) => {
    setAdminCategory(newCat);
    if (newCat === "Fit" && adminPercentage < 80) {
      setAdminPercentage(88.0);
    } else if (newCat === "Close Fit" && (adminPercentage < 60 || adminPercentage >= 80)) {
      setAdminPercentage(72.0);
    } else if (newCat === "Not Fit" && adminPercentage >= 60) {
      setAdminPercentage(25.0);
    }
  };

  const handleAdminPercentageChange = (valStr: string) => {
    const num = parseFloat(valStr);
    if (Number.isNaN(num)) {
      setAdminPercentage(0);
      return;
    }
    const clamped = Math.max(0, Math.min(100, Number(num.toFixed(2))));
    setAdminPercentage(clamped);
    if (clamped >= 80) {
      setAdminCategory("Fit");
    } else if (clamped >= 60) {
      setAdminCategory("Close Fit");
    } else {
      setAdminCategory("Not Fit");
    }
  };

  // Click [Validasi] -> Save to CSV + Retrain Model
  const handleValidateAndRetrain = async () => {
    if (!result || !token) return;
    setValidating(true);
    setValidationMessage(null);
    try {
      const resp = await fetch(`${API_BASE}/matching/validate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          analysis_id: result.id,
          cv_filename: result.cv_filename,
          cv_text: result.cv_text_full || result.cv_text_preview,
          job_title: jobTitle,
          job_description: jobDescription,
          validated_category: adminCategory,
          validated_percentage: adminPercentage,
          trigger_retrain: true,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Gagal memvalidasi hasil analisis.");
      }

      setResult({
        ...result,
        status_category: data.validated_category,
        similarity_percentage: data.validated_percentage,
        similarity_score: data.validated_score,
        is_validated: true,
      });
      setValidationMessage(data.message);
      if (data.retrain_metrics) {
        setRetrainMetrics(data.retrain_metrics);
      }
      fetchAdminStats(token);
    } catch (err: any) {
      setAnalyzeError(err.message || "Terjadi kesalahan saat validasi dan re-train.");
    } finally {
      setValidating(false);
    }
  };

  const handleStandaloneRetrain = async () => {
    if (!token) return;
    setRetrainingStandalone(true);
    setValidationMessage(null);
    try {
      const resp = await fetch(`${API_BASE}/matching/retrain`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ epochs: 4 }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.detail || "Gagal melakukan re-training model.");
      }
      setValidationMessage(data.message);
      if (data.retrain_metrics) {
        setRetrainMetrics(data.retrain_metrics);
      }
      fetchAdminStats(token);
    } catch (err: any) {
      setAnalyzeError(err.message || "Gagal menjalankan re-train model.");
    } finally {
      setRetrainingStandalone(false);
    }
  };

  const getStatusClass = (status: string) => {
    if (status === "Fit") return "status-badge status-fit";
    if (status === "Close Fit") return "status-badge status-close";
    return "status-badge status-not";
  };

  // Compute SVG Donut segments for Admin Processing Summary Chart
  const renderDonutChart = (percentages: Record<string, number>, total: number) => {
    const fitPct = percentages["Fit"] || 0;
    const closePct = percentages["Close Fit"] || 0;
    const notPct = percentages["Not Fit"] || 0;

    const radius = 62;
    const circumference = 2 * Math.PI * radius;
    const fitDash = (fitPct / 100) * circumference;
    const closeDash = (closePct / 100) * circumference;
    const notDash = (notPct / 100) * circumference;

    return (
      <div style={{ display: "flex", alignItems: "center", gap: "1.5rem", flexWrap: "wrap" }}>
        <svg width="160" height="160" viewBox="0 0 160 160" aria-label="Grafik Persentase Hasil Pemrosesan">
          <circle cx="80" cy="80" r={radius} fill="transparent" stroke="#e2e8f0" strokeWidth="22" />
          {total > 0 && (
            <>
              {/* Fit segment (Green) */}
              <circle
                cx="80"
                cy="80"
                r={radius}
                fill="transparent"
                stroke="#16a34a"
                strokeWidth="22"
                strokeDasharray={`${fitDash} ${circumference - fitDash}`}
                strokeDashoffset={circumference * 0.25}
              />
              {/* Close Fit segment (Amber) */}
              <circle
                cx="80"
                cy="80"
                r={radius}
                fill="transparent"
                stroke="#d97706"
                strokeWidth="22"
                strokeDasharray={`${closeDash} ${circumference - closeDash}`}
                strokeDashoffset={circumference * 0.25 - fitDash}
              />
              {/* Not Fit segment (Red) */}
              <circle
                cx="80"
                cy="80"
                r={radius}
                fill="transparent"
                stroke="#dc2626"
                strokeWidth="22"
                strokeDasharray={`${notDash} ${circumference - notDash}`}
                strokeDashoffset={circumference * 0.25 - fitDash - closeDash}
              />
            </>
          )}
          <text x="80" y="76" textAnchor="middle" fontSize="20" fontWeight="700" fill="#0f172a">
            {total}
          </text>
          <text x="80" y="95" textAnchor="middle" fontSize="11" fill="#64748b">
            Total CV
          </text>
        </svg>

        <div style={{ flex: 1, minWidth: "200px", display: "flex", flexDirection: "column", gap: "0.65rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.88rem", fontWeight: 600 }}>
              <span style={{ width: 12, height: 12, borderRadius: 3, background: "#16a34a", display: "inline-block" }} />
              Fit (&ge; 80%)
            </span>
            <span style={{ fontWeight: 700, color: "#15803d" }}>{fitPct.toFixed(1)}%</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.88rem", fontWeight: 600 }}>
              <span style={{ width: 12, height: 12, borderRadius: 3, background: "#d97706", display: "inline-block" }} />
              Close Fit (60% – 79.9%)
            </span>
            <span style={{ fontWeight: 700, color: "#b45309" }}>{closePct.toFixed(1)}%</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.88rem", fontWeight: 600 }}>
              <span style={{ width: 12, height: 12, borderRadius: 3, background: "#dc2626", display: "inline-block" }} />
              Not Fit (&lt; 60%)
            </span>
            <span style={{ fontWeight: 700, color: "#b91c1c" }}>{notPct.toFixed(1)}%</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <>
      {/* TOP BAR */}
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <div className="brand-badge">mB</div>
            <div>
              <div className="brand-title">CV &ndash; Job Matching System (mBERT + Cosine Similarity)</div>
              <div className="brand-sub">
                Deep Learning-Based NER (EDU, ORG, SKILL, ROLE, DUR) &bull; Siamese Multilingual BERT
              </div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            {isAdmin ? (
              <>
                <span className="role-pill role-pill-admin">Mode Admin ({adminUsername})</span>
                <button
                  type="button"
                  className="btn btn-outline"
                  onClick={handleStandaloneRetrain}
                  disabled={retrainingStandalone}
                >
                  {retrainingStandalone ? "Re-training..." : "Re-train Model"}
                </button>
                <button type="button" className="btn btn-danger-outline" onClick={handleLogout}>
                  Logout
                </button>
              </>
            ) : (
              <>
                <span className="role-pill role-pill-user">Mode: User Biasa</span>
                <button type="button" className="btn btn-primary" onClick={openLoginModal}>
                  Sign in
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* NATIVE MODAL DIALOG FOR ADMIN LOGIN */}
      <dialog
        ref={loginDialogRef}
        aria-labelledby="login-dialog-title"
        {...({ closedby: "any" } as any)}
      >
        <div style={{ padding: "1.5rem" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "1rem",
              paddingBottom: "0.75rem",
              borderBottom: "1px solid var(--border)",
            }}
          >
            <h2 id="login-dialog-title" style={{ fontSize: "1.1rem", fontWeight: 700 }}>
              Login User Admin
            </h2>
            <button
              type="button"
              className="btn btn-outline"
              style={{ padding: "0.3rem 0.65rem", fontSize: "0.8rem" }}
              onClick={closeLoginModal}
              aria-label="Tutup dialog login"
            >
              Tutup
            </button>
          </div>

          <form onSubmit={handleLoginSubmit}>
            <div className="form-group">
              <label htmlFor="username" className="label">
                Username
              </label>
              <input
                id="username"
                name="username"
                type="text"
                className="input"
                value={usernameInput}
                onChange={(e) => setUsernameInput(e.target.value)}
                autoComplete="username"
                spellCheck={false}
                autoCorrect="off"
                autoCapitalize="none"
                enterKeyHint="next"
                required
                autoFocus
              />
            </div>

            <div className="form-group">
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <label htmlFor="current-password" className="label">
                  Password
                </label>
                <button
                  type="button"
                  onClick={() => setShowPassword((prev) => !prev)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--primary)",
                    fontSize: "0.78rem",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {showPassword ? "Sembunyikan" : "Tampilkan password"}
                </button>
              </div>
              <input
                id="current-password"
                name="password"
                type={showPassword ? "text" : "password"}
                className="input"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                autoComplete="current-password"
                enterKeyHint="done"
                placeholder="Masukkan password admin..."
                required
              />
            </div>

            {loginError && <div className="alert alert-error">{loginError}</div>}

            <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.65rem", marginTop: "1rem" }}>
              <button type="button" className="btn btn-outline" onClick={closeLoginModal}>
                Batal
              </button>
              <button type="submit" className="btn btn-primary" disabled={loginLoading}>
                {loginLoading ? "Memverifikasi..." : "Login"}
              </button>
            </div>
          </form>
        </div>
      </dialog>

      {/* MAIN CONTENT */}
      <main className="container">
        <div className="grid-2">
          {/* LEFT COLUMN: UPLOAD CV & JOB INPUT */}
          <section className="card">
            <div className="card-header">
              <div>
                <h1 className="card-title">1. Upload CV &amp; Pilih Lowongan Pekerjaan</h1>
                <p className="card-subtitle">
                  Unggah dokumen CV (.pdf) dan tentukan kriteria lowongan pekerjaan untuk dianalisis oleh mBERT
                </p>
              </div>
            </div>

            <form onSubmit={handleAnalyzeSubmit}>
              <div className="form-group">
                <label htmlFor="preset-job-select" className="label">
                  Pilih Preset Lowongan (Dataset) atau Ketik Manual di Bawah
                </label>
                <select
                  id="preset-job-select"
                  className="select"
                  value={selectedPresetId}
                  onChange={handlePresetChange}
                >
                  {presetJobs.map((job) => (
                    <option key={job.id} value={job.id}>
                      {job.job_title} ({job.domain})
                    </option>
                  ))}
                  <option value="custom">-- Ketik / Edit Lowongan Manual --</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="job-title-input" className="label">
                  Judul Posisi Lowongan (Job Title)
                </label>
                <input
                  id="job-title-input"
                  type="text"
                  className="input"
                  value={jobTitle}
                  onChange={(e) => {
                    setJobTitle(e.target.value);
                    setSelectedPresetId("custom");
                  }}
                  placeholder="Contoh: Consumer Banking Operations Manager"
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="job-desc-input" className="label">
                  Deskripsi &amp; Kualifikasi Lowongan (Job Description)
                </label>
                <textarea
                  id="job-desc-input"
                  className="textarea"
                  rows={4}
                  value={jobDescription}
                  onChange={(e) => {
                    setJobDescription(e.target.value);
                    setSelectedPresetId("custom");
                  }}
                  placeholder="Masukkan kualifikasi pendidikan, pengalaman kerja, dan keahlian yang dibutuhkan..."
                  required
                />
              </div>

              {/* Exact Whiteboard Row: Upload CV [Choose File] [Analisis] */}
              <div
                style={{
                  background: "var(--bg-muted)",
                  border: "1px dashed var(--border-strong)",
                  borderRadius: "var(--radius-md)",
                  padding: "1rem",
                  marginTop: "0.5rem",
                }}
              >
                <label htmlFor="cv-file-input" className="label" style={{ display: "block", marginBottom: "0.5rem" }}>
                  Upload CV Pelamar (.pdf)
                </label>
                <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.75rem" }}>
                  <input
                    id="cv-file-input"
                    type="file"
                    accept=".pdf,application/pdf"
                    className="input"
                    style={{ flex: 1, minWidth: "220px", padding: "0.45rem" }}
                    onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                    required
                  />
                  <button type="submit" className="btn btn-primary" disabled={analyzing}>
                    {analyzing ? "Menganalisis..." : "Analisis"}
                  </button>
                </div>
                {selectedFile && (
                  <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: "0.45rem" }}>
                    Berkas terpilih: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024).toFixed(1)} KB)
                  </div>
                )}
              </div>

              {analyzeError && <div className="alert alert-error">{analyzeError}</div>}
            </form>
          </section>

          {/* RIGHT COLUMN: HASIL ANALISIS (WHITEBOARD DESIGN FOR ADMIN VS USER BIASA) */}
          <section className="card">
            <div className="card-header">
              <div>
                <h2 className="card-title">2. Hasil Analisis Kecocokan (mBERT + Cosine)</h2>
                <p className="card-subtitle">
                  {isAdmin
                    ? "Mode Admin: Anda dapat mengoreksi kategori (Combobox) & nilai Cosine (Numeric Up-Down), lalu klik Validasi untuk menyimpan ke CSV & Re-train model."
                    : "Mode User Biasa: Menampilkan hasil klasifikasi dan persentase kecocokan secara langsung (teks biasa)."}
                </p>
              </div>
            </div>

            {!result ? (
              <div
                style={{
                  textAlign: "center",
                  padding: "2.5rem 1rem",
                  color: "var(--text-muted)",
                  border: "1px dashed var(--border)",
                  borderRadius: "var(--radius-md)",
                }}
              >
                <p style={{ fontWeight: 600, color: "var(--text-secondary)", marginBottom: "0.25rem" }}>
                  Belum ada CV yang dianalisis
                </p>
                <p style={{ fontSize: "0.84rem" }}>
                  Pilih berkas PDF CV di sebelah kiri lalu klik tombol <strong>Analisis</strong>.
                </p>
              </div>
            ) : (
              <div>
                {/* WHITEBOARD HASIL ANALISIS BOX */}
                <div className="whiteboard-result-box">
                  <div
                    style={{
                      fontSize: "0.8rem",
                      fontWeight: 700,
                      textTransform: "uppercase",
                      letterSpacing: "0.04em",
                      color: "var(--text-muted)",
                    }}
                  >
                    Hasil Analisis ({result.cv_filename} &rarr; {result.job_title})
                  </div>

                  {isAdmin ? (
                    /* ADMIN VIEW: Combobox [Fit | v] + Numeric Up-Down ([ 97.06 ] ^v %) + [Validasi] */
                    <div>
                      <div className="whiteboard-controls-row">
                        <div>
                          <label
                            htmlFor="admin-category-combobox"
                            style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.2rem" }}
                          >
                            Kategori (Combobox)
                          </label>
                          <select
                            id="admin-category-combobox"
                            className="select"
                            style={{ width: "165px", fontWeight: 700, fontSize: "1rem" }}
                            value={adminCategory}
                            onChange={(e) => handleAdminCategoryChange(e.target.value)}
                          >
                            <option value="Fit">Fit</option>
                            <option value="Close Fit">Close Fit</option>
                            <option value="Not Fit">Not Fit</option>
                          </select>
                        </div>

                        <div>
                          <label
                            htmlFor="admin-cosine-updown"
                            style={{ display: "block", fontSize: "0.75rem", fontWeight: 600, marginBottom: "0.2rem" }}
                          >
                            Nilai Cosine (Numeric Up-Down)
                          </label>
                          <div style={{ display: "flex", alignItems: "center", gap: "0.35rem" }}>
                            <span style={{ fontSize: "1.2rem", fontWeight: 600 }}>(</span>
                            <input
                              id="admin-cosine-updown"
                              type="number"
                              step="0.1"
                              min="0"
                              max="100"
                              className="input"
                              style={{ width: "115px", fontWeight: 700, fontSize: "1rem", textAlign: "right" }}
                              value={adminPercentage}
                              onChange={(e) => handleAdminPercentageChange(e.target.value)}
                            />
                            <span style={{ fontSize: "1.1rem", fontWeight: 700 }}>% )</span>
                          </div>
                        </div>

                        <div style={{ alignSelf: "flex-end" }}>
                          <button
                            type="button"
                            className="btn btn-success"
                            onClick={handleValidateAndRetrain}
                            disabled={validating}
                          >
                            {validating ? "Memvalidasi & Re-train..." : "Validasi"}
                          </button>
                        </div>
                      </div>

                      <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)", marginTop: "0.65rem" }}>
                        Klik <strong>Validasi</strong> untuk menyimpan CV ini ke{" "}
                        <code>dataset_cv_job_matching_3class_1500.csv</code> sebagai data training baru dan langsung
                        melakukan <em>re-train</em> model Siamese mBERT.
                      </div>
                    </div>
                  ) : (
                    /* USER BIASA VIEW: Plain Text Only (Tanpa Combobox, Tanpa Numeric Up-Down, Tanpa Tombol Validasi) */
                    <div style={{ marginTop: "0.65rem" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.85rem", flexWrap: "wrap" }}>
                        <span className={getStatusClass(result.status_category)} style={{ fontSize: "1.15rem" }}>
                          {result.status_category}
                        </span>
                        <span style={{ fontSize: "1.45rem", fontWeight: 800, color: "var(--text-primary)" }}>
                          ({result.similarity_percentage.toFixed(2)}%)
                        </span>
                      </div>
                      <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
                        Ambang Batas Keputusan: <strong>Fit</strong> (&ge; 80%) &bull; <strong>Close Fit</strong> (60%
                        &ndash; 79.9%) &bull; <strong>Not Fit</strong> (&lt; 60%)
                      </div>
                    </div>
                  )}

                  {validationMessage && <div className="alert alert-success">{validationMessage}</div>}

                  {retrainMetrics && isAdmin && (
                    <div
                      style={{
                        marginTop: "0.75rem",
                        padding: "0.75rem",
                        background: "#eff6ff",
                        border: "1px solid #93c5fd",
                        borderRadius: "8px",
                        fontSize: "0.8rem",
                        color: "#1e3a8a",
                      }}
                    >
                      <strong>Ringkasan Re-Train Model ({retrainMetrics.model_checkpoint}):</strong>
                      <div style={{ marginTop: "0.25rem" }}>
                        Epoch: <strong>{retrainMetrics.epochs_trained}</strong> &bull; Sampel Batch:{" "}
                        <strong>{retrainMetrics.samples_trained}</strong> &bull; Loss:{" "}
                        <strong>{retrainMetrics.initial_loss}</strong> &rarr;{" "}
                        <strong>{retrainMetrics.final_loss}</strong> &bull; Durasi:{" "}
                        <strong>{retrainMetrics.duration_seconds}s</strong> &bull; Total Dataset CSV:{" "}
                        <strong>{retrainMetrics.total_dataset_rows} baris</strong>
                      </div>
                    </div>
                  )}
                </div>

                {/* NER ENTITIES TABLE 2.5 SKRIPSI */}
                <div>
                  <h3 style={{ fontSize: "0.92rem", fontWeight: 700, marginBottom: "0.6rem" }}>
                    Deteksi Entitas Kompetensi NER mBERT (Tabel 2.5 Skripsi)
                  </h3>
                  <div style={{ display: "grid", gap: "0.55rem", fontSize: "0.84rem" }}>
                    <div>
                      <strong>ROLE (Jabatan): </strong>
                      {result.extracted_entities.ROLE?.length ? (
                        result.extracted_entities.ROLE.map((item, i) => (
                          <span key={i} className="entity-chip">
                            {item}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>Tidak terdeteksi</span>
                      )}
                    </div>
                    <div>
                      <strong>EDU (Pendidikan): </strong>
                      {result.extracted_entities.EDU?.length ? (
                        result.extracted_entities.EDU.map((item, i) => (
                          <span key={i} className="entity-chip">
                            {item}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>Tidak terdeteksi</span>
                      )}
                    </div>
                    <div>
                      <strong>ORG (Organisasi / Institusi): </strong>
                      {result.extracted_entities.ORG?.length ? (
                        result.extracted_entities.ORG.map((item, i) => (
                          <span key={i} className="entity-chip">
                            {item}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>Tidak terdeteksi</span>
                      )}
                    </div>
                    <div>
                      <strong>DUR (Durasi Pengalaman): </strong>
                      {result.extracted_entities.DUR?.length ? (
                        result.extracted_entities.DUR.map((item, i) => (
                          <span key={i} className="entity-chip">
                            {item}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>Tidak terdeteksi</span>
                      )}
                    </div>
                    <div>
                      <strong>SKILL (Keahlian Kompetensi): </strong>
                      {result.extracted_entities.SKILL?.length ? (
                        result.extracted_entities.SKILL.map((item, i) => (
                          <span key={i} className="entity-chip">
                            {item}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>Tidak terdeteksi</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>

        {/* ADMIN-ONLY SECTION: SUMMARY CHARTS & VALIDATION HISTORY */}
        {isAdmin && stats && (
          <section className="card">
            <div className="card-header">
              <div>
                <h2 className="card-title">
                  3. Chart Rangkuman Hasil Pemrosesan &amp; Distribusi Dataset (Khusus Admin)
                </h2>
                <p className="card-subtitle">
                  Visualisasi persentase klasifikasi <strong>Fit</strong>, <strong>Close Fit</strong>, dan{" "}
                  <strong>Not Fit</strong> dari riwayat pemrosesan CV serta file dataset pelatihan CSV
                </p>
              </div>
            </div>

            <div className="grid-2">
              {/* CHART 1: RIWAYAT PEMROSESAN CV DI APLIKASI */}
              <div
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-md)",
                  padding: "1.25rem",
                  background: "var(--bg-muted)",
                }}
              >
                <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                  A. Persentase Hasil Pemrosesan CV ({stats.history_summary.total_processed} Dokumen)
                </h3>
                <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
                  Tervalidasi oleh Admin: <strong>{stats.history_summary.validated_count}</strong> dokumen
                </p>
                {renderDonutChart(stats.history_summary.percentages, stats.history_summary.total_processed)}
              </div>

              {/* CHART 2: DISTRIBUSI DATASET CSV PELATIHAN */}
              <div
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-md)",
                  padding: "1.25rem",
                  background: "var(--bg-muted)",
                }}
              >
                <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: "0.25rem" }}>
                  B. Distribusi Dataset CSV ({stats.dataset_summary.dataset_filename})
                </h3>
                <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginBottom: "1rem" }}>
                  Total Baris: <strong>{stats.dataset_summary.total_rows}</strong> (Data Baru Hasil Validasi:{" "}
                  <strong>+{stats.dataset_summary.newly_validated_rows}</strong> baris)
                </p>

                <div style={{ display: "flex", flexDirection: "column", gap: "0.85rem" }}>
                  {(["Fit", "Close Fit", "Not Fit"] as const).map((cat) => {
                    const count = stats.dataset_summary.counts[cat] || 0;
                    const pct = stats.dataset_summary.percentages[cat] || 0;
                    const barColor = cat === "Fit" ? "#16a34a" : cat === "Close Fit" ? "#d97706" : "#dc2626";
                    return (
                      <div key={cat}>
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            fontSize: "0.84rem",
                            fontWeight: 600,
                            marginBottom: "0.3rem",
                          }}
                        >
                          <span>{cat}</span>
                          <span>
                            {count} baris ({pct.toFixed(2)}%)
                          </span>
                        </div>
                        <div
                          style={{
                            height: "14px",
                            background: "#e2e8f0",
                            borderRadius: "999px",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${Math.min(100, pct)}%`,
                              height: "100%",
                              background: barColor,
                              transition: "width 0.3s ease",
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* RECENT PROCESSED CV TABLE */}
            {stats.recent_history.length > 0 && (
              <div style={{ marginTop: "1.5rem" }}>
                <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: "0.75rem" }}>
                  Riwayat Analisis &amp; Status Validasi Terbaru
                </h3>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Berkas CV</th>
                        <th>Lowongan Pekerjaan</th>
                        <th>Nilai Cosine (%)</th>
                        <th>Kategori</th>
                        <th>Status Validasi CSV</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stats.recent_history.slice(0, 10).map((item) => (
                        <tr key={item.id}>
                          <td style={{ fontWeight: 600 }}>{item.cv_filename}</td>
                          <td>{item.job_title}</td>
                          <td style={{ fontWeight: 700 }}>{item.similarity_percentage.toFixed(2)}%</td>
                          <td>
                            <span className={getStatusClass(item.status_category)} style={{ fontSize: "0.78rem" }}>
                              {item.status_category}
                            </span>
                          </td>
                          <td>
                            {item.is_validated ? (
                              <span style={{ color: "#15803d", fontWeight: 600 }}>
                                &#10003; Tersimpan di CSV &amp; Tervalidasi
                              </span>
                            ) : (
                              <span style={{ color: "var(--text-muted)" }}>Belum divalidasi</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </>
  );
}

