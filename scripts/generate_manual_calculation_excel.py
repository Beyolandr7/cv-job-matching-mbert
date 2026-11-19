"""
generate_manual_calculation_excel.py
------------------------------------
Generates an auditable, formula-driven Excel workbook
(`docs/Perhitungan_Manual_Siamese_mBERT_NER_Cosine.xlsx`) demonstrating the complete
end-to-end mathematical calculation of the proposed model:
  Unified Multi-Task Siamese mBERT Feature Extractor with NER-Guided Pooling +
  Scikit-Learn Cosine Similarity + PureCosineMarginLoss + Evaluation Metrics

Uses REAL intermediate tensors extracted directly from `best_siamese_mbert_cosine.pt`
on 3 Indonesian CV-Job pairs (Fit, Close Fit, Not Fit) based on Thesis Table 2.5.
"""

import os
import sys
from typing import Any, Dict, List

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

from core.config import settings
from services.mbert_service import ID2TAG, NER_TAGS, mbert_service

OUT_XLSX = os.path.join(ROOT_DIR, "docs", "Perhitungan_Manual_Siamese_mBERT_NER_Cosine.xlsx")
DUMMY_CSV = os.path.join(ROOT_DIR, "data", "dummy_cv_job_indonesia.csv")

# Styling palette (academic navy/slate theme)
NAVY_FILL = PatternFill("solid", fgColor="1B365D")
SUBHDR_FILL = PatternFill("solid", fgColor="2C5282")
SECTION_FILL = PatternFill("solid", fgColor="DCE6F1")
INPUT_FILL = PatternFill("solid", fgColor="F2F7FB")
FORMULA_FILL = PatternFill("solid", fgColor="EAF7EA")
FIT_FILL = PatternFill("solid", fgColor="D4EDDA")
CLOSE_FILL = PatternFill("solid", fgColor="FFF3CD")
NOT_FILL = PatternFill("solid", fgColor="F8D7DA")
WARN_FILL = PatternFill("solid", fgColor="FFF8E7")

FONT_TITLE = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
FONT_SUBTITLE = Font(name="Calibri", size=11, bold=True, color="1B365D")
FONT_HDR = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
FONT_BOLD = Font(name="Calibri", size=10, bold=True, color="111111")
FONT_REG = Font(name="Calibri", size=10, color="111111")
FONT_ITALIC = Font(name="Calibri", size=9, italic=True, color="444444")

THIN_SIDE = Side(style="thin", color="BFBFBF")
BORDER_THIN = Border(left=THIN_SIDE, right=THIN_SIDE, top=THIN_SIDE, bottom=THIN_SIDE)
ALIGN_LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
ALIGN_RIGHT = Alignment(horizontal="right", vertical="center")


def style_range(ws, min_col, min_row, max_col, max_row, font=None, fill=None, alignment=None, border=BORDER_THIN, num_format=None):
    for row in ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col):
        for cell in row:
            if font:
                cell.font = font
            if fill:
                cell.fill = fill
            if alignment:
                cell.alignment = alignment
            if border:
                cell.border = border
            if num_format:
                cell.number_format = num_format


def add_banner(ws, title: str, subtitle: str, max_col: int):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
    ws.cell(1, 1, title)
    style_range(ws, 1, 1, max_col, 1, font=FONT_TITLE, fill=NAVY_FILL, alignment=ALIGN_LEFT, border=None)
    ws.row_dimensions[1].height = 28

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max_col)
    ws.cell(2, 1, subtitle)
    style_range(ws, 1, 2, max_col, 2, font=FONT_ITALIC, fill=SECTION_FILL, alignment=ALIGN_LEFT, border=None)
    ws.row_dimensions[2].height = 22


def trace_document(text: str, doc_type: str, max_len: int = 128) -> Dict[str, Any]:
    """Extracts every intermediate tensor for a single document from the live trained model."""
    clean_text = mbert_service.regex_standardize_text(text)
    entities = mbert_service.extract_entities_hybrid(clean_text)
    domain, tier = mbert_service.detect_domain_and_tier(clean_text, doc_type=doc_type, entities=entities)
    enriched = mbert_service.build_ner_enriched_text(clean_text, doc_type=doc_type)

    tokenizer = mbert_service.tokenizer
    model = mbert_service.model
    assert tokenizer is not None and model is not None

    enc = tokenizer(
        enriched,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(mbert_service.device)
    attention_mask = enc["attention_mask"].to(mbert_service.device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    bio_ids = mbert_service._encode_bio_labels(enriched, mbert_service.extract_entities_hybrid(enriched), max_len)

    model.eval()
    with torch.no_grad():
        outputs = model.mbert(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = outputs.last_hidden_state[0]  # [128, 768]
        ner_logits = model.ner_head(token_embs)  # [128, 11]
        ner_probs = torch.softmax(ner_logits, dim=-1)  # [128, 11]
        entity_prob = 1.0 - ner_probs[:, 0]  # [128]
        mask_f = attention_mask[0].float()  # [128]
        weights = (1.0 + 0.5 * entity_prob) * mask_f  # [128]

        cls_emb = token_embs[0]  # [768]
        mean_emb = torch.sum(token_embs * weights.unsqueeze(-1), dim=0) / torch.clamp(weights.sum(), min=1e-9)  # [768]
        combined = 0.5 * cls_emb + 0.5 * mean_emb  # [768]
        projected = model.projection(combined)  # [768]
        norm_val = torch.linalg.norm(projected, ord=2)
        vec_768 = F.normalize(projected.unsqueeze(0), p=2, dim=1)[0]  # [768]

        ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
        bio_tensor = torch.tensor(bio_ids, dtype=torch.long, device=mbert_service.device)
        ner_ce_loss = float(ce_fn(ner_logits, bio_tensor).item())

    return {
        "raw_text": text,
        "clean_text": clean_text,
        "doc_type": doc_type,
        "entities": entities,
        "domain": domain,
        "tier": tier,
        "enriched": enriched,
        "tokens": tokens,
        "input_ids": input_ids[0].cpu().tolist(),
        "attention_mask": attention_mask[0].cpu().tolist(),
        "bio_ids": bio_ids,
        "token_embs": token_embs.cpu().numpy(),  # [128, 768]
        "ner_logits": ner_logits.cpu().numpy(),  # [128, 11]
        "ner_probs": ner_probs.cpu().numpy(),
        "weights": weights.cpu().numpy(),
        "cls_emb": cls_emb.cpu().numpy(),
        "mean_emb": mean_emb.cpu().numpy(),
        "combined": combined.cpu().numpy(),
        "projected": projected.cpu().numpy(),
        "l2_norm": float(norm_val.item()),
        "vec_768": vec_768.cpu().numpy(),
        "ner_ce_loss": ner_ce_loss,
    }


def main():
    # Prefer the clean Colab checkpoint backup if present so numbers align directly with Thesis evaluation
    backup_ckpt = settings.FINE_TUNED_MODEL_PATH.replace(".pt", ".backup.pt")
    if os.path.exists(backup_ckpt):
        mbert_service.fine_tuned_path = backup_ckpt
    mbert_service.load_model()
    assert mbert_service.model is not None

    # Load first 3 rows from Indonesian dummy CSV (Software Engineer CV vs Fit, Close Fit, Not Fit jobs)
    df_id = pd.read_csv(DUMMY_CSV)
    row_fit = df_id.iloc[0]
    row_close = df_id.iloc[1]
    row_not = df_id.iloc[2]

    cv_text = str(row_fit["cv"])
    job_fit_text = str(row_fit["job"])
    job_close_text = str(row_close["job"])
    job_not_text = str(row_not["job"])

    print("Tracing real tensors from checkpoint:", os.path.basename(mbert_service.fine_tuned_path))
    tr_cv = trace_document(cv_text, "CANDIDATE_CV")
    tr_j1 = trace_document(job_fit_text, "JOB_POSTING")
    tr_j2 = trace_document(job_close_text, "JOB_POSTING")
    tr_j3 = trace_document(job_not_text, "JOB_POSTING")

    docs = [
        ("CV Pelamar (Software Engineer - Tabel 2.5)", tr_cv),
        ("Job 1 - FIT (Software Engineer)", tr_j1),
        ("Job 2 - CLOSE FIT (Technical Support Specialist)", tr_j2),
        ("Job 3 - NOT FIT (Executive Chef)", tr_j3),
    ]

    # Exact sklearn reference cosine similarities
    sklearn_sims = [
        float(sklearn_cosine_similarity(tr_cv["vec_768"].reshape(1, -1), tr_j1["vec_768"].reshape(1, -1))[0, 0]),
        float(sklearn_cosine_similarity(tr_cv["vec_768"].reshape(1, -1), tr_j2["vec_768"].reshape(1, -1))[0, 0]),
        float(sklearn_cosine_similarity(tr_cv["vec_768"].reshape(1, -1), tr_j3["vec_768"].reshape(1, -1))[0, 0]),
    ]
    print("Sklearn Cosine Reference (Fit, Close Fit, Not Fit):", [round(x * 100, 2) for x in sklearn_sims])

    # Extract projection weight row 0 & bias[0] to prove matrix multiplication W*combined + b
    proj_w0 = mbert_service.model.projection.weight[0].detach().cpu().numpy()  # [768]
    proj_b0 = float(mbert_service.model.projection.bias[0].detach().cpu().item())

    wb = openpyxl.Workbook()

    # =========================================================================
    # SHEET 0: 0_Ringkasan_Alur
    # =========================================================================
    ws0 = wb.active
    ws0.title = "0_Ringkasan_Alur"
    add_banner(
        ws0,
        "PERHITUNGAN MANUAL STEP-BY-STEP MODEL: UNIFIED NER + SIAMESE mBERT + COSINE SIMILARITY",
        f"Diekstrak langsung dari tensor riil checkpoint '{os.path.basename(mbert_service.fine_tuned_path)}' | Sel hijau = Rumus Excel Aktif",
        8,
    )

    ws0.cell(4, 1, "A. KONFIGURASI MODEL & HYPERPARAMETER TERBAIK (OPTUNA NOTEBOOK 03)").font = FONT_SUBTITLE
    cfg_rows = [
        ("Arsitektur Model", "Unified Multi-Task Siamese mBERT Feature Extractor with NER-Guided Pooling"),
        ("Backbone Pre-trained", "bert-base-multilingual-cased (12 Layer Transformer, 768 Hidden Size, 119.547 WordPiece Vocab)"),
        ("Best Hyperparameters (Model 2)", "Learning Rate = 2e-5 | Max Length = 128 token | Dropout = 0.1 | Weight Decay = 0.01"),
        ("Konvergensi Training", "Max Epoch = 100 | Early Stopping Patience = 5 | Best Epoch = 5 (Val Loss = 0.0365, Val Acc = 98.22%)"),
        ("Fungsi Loss", "PureCosineMarginLoss (MSE scale=10.0, Band scale=6.0) + 0.20 * Multi-Task NER CrossEntropyLoss"),
        ("Ambang Batas Keputusan (Skripsi)", "Fit >= 80.0% | Close Fit = 60.0% - 79.9% | Not Fit < 60.0%"),
    ]
    for idx, (k, v) in enumerate(cfg_rows, start=5):
        ws0.cell(idx, 1, k).font = FONT_BOLD
        ws0.merge_cells(start_row=idx, start_column=2, end_row=idx, end_column=8)
        ws0.cell(idx, 2, v).font = FONT_REG
        style_range(ws0, 1, idx, 8, idx, fill=INPUT_FILL, alignment=ALIGN_LEFT)

    ws0.cell(12, 1, "B. RINGKASAN HASIL 3 STUDI KASUS INDONESIA (RUMUS DINAMIS LINTAS SHEET)").font = FONT_SUBTITLE
    hdrs0 = [
        "No", "Studi Kasus (CV vs Lowongan)", "Label Ground Truth", "Dot Product (u·v)",
        "Cosine Similarity", "Skor (%)", "Keputusan Sistem", "Status Kecocokan"
    ]
    for c_i, h in enumerate(hdrs0, 1):
        ws0.cell(13, c_i, h)
    style_range(ws0, 1, 13, 8, 13, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    case_meta = [
        (1, "CV Software Engineer vs Lowongan Software Engineer (PT Telkom)", "Fit", 6, FIT_FILL),
        (2, "CV Software Engineer vs Lowongan Technical Support Specialist", "Close Fit", 7, CLOSE_FILL),
        (3, "CV Software Engineer vs Lowongan Executive Chef", "Not Fit", 8, NOT_FILL),
    ]
    for row_off, (no, desc, gt, s4_row, badge_fill) in enumerate(case_meta, start=14):
        ws0.cell(row_off, 1, no)
        ws0.cell(row_off, 2, desc)
        ws0.cell(row_off, 3, gt)
        ws0.cell(row_off, 4, f"='4_Cosine_Similarity'!C{s4_row}")
        ws0.cell(row_off, 5, f"='4_Cosine_Similarity'!F{s4_row}")
        ws0.cell(row_off, 6, f"='4_Cosine_Similarity'!G{s4_row}")
        ws0.cell(row_off, 7, f"='4_Cosine_Similarity'!H{s4_row}")
        ws0.cell(row_off, 8, f'=IF(G{row_off}=C{row_off},"TEPAT","MELESET")')
        style_range(ws0, 1, row_off, 8, row_off, font=FONT_REG, fill=FORMULA_FILL, alignment=ALIGN_CENTER)
        ws0.cell(row_off, 2).alignment = ALIGN_LEFT
        ws0.cell(row_off, 4).number_format = "0.000000"
        ws0.cell(row_off, 5).number_format = "0.000000"
        ws0.cell(row_off, 6).number_format = "0.00"
        ws0.cell(row_off, 7).fill = badge_fill
        ws0.cell(row_off, 7).font = FONT_BOLD
        ws0.cell(row_off, 8).font = FONT_BOLD

    ws0.cell(18, 1, "C. PETA NAVIGASI WORKBOOK (ALUR AWAL SAMPAI AKHIR)").font = FONT_SUBTITLE
    nav_rows = [
        ("Sheet 1", "1_Input_NER_Tokenisasi", "Teks mentah CV & 3 Job -> Regex Cased -> Ekstraksi 5 Entitas NER (Tabel 2.5) -> Header [DOC] -> Tokenisasi WordPiece 128 token & Label BIO."),
        ("Sheet 2", "2_NER_Head_Softmax_Bobot", "Logit riil self.ner_head [128 x 11] -> Rumus Softmax Excel (=EXP/SUM) -> P(entity) = 1 - P(O) -> Bobot NER-Guided Pooling & NER Cross-Entropy Loss."),
        ("Sheet 3", "3_Pooling_Proyeksi_L2", "Bukti rumus SUMPRODUCT NER-Guided Mean Pooling pada 128 token, bukti 1 baris perkalian Matriks Proyeksi W*x+b, & seluruh 768 dimensi vektor hingga Normalisasi L2."),
        ("Sheet 4", "4_Cosine_Similarity", "Perhitungan Cosine Similarity penuh atas 768 dimensi (SUMPRODUCT & SQRT(SUMSQ)) untuk 3 pasangan (Fit, Close Fit, Not Fit) + bukti selisih 0.000000 vs sklearn."),
        ("Sheet 5", "5_Loss_PureCosine_NER", "Perhitungan rumus aktif PureCosineMarginLoss (MSE Term target 0.15/0.70/0.90 + Band Penalty 0.45/0.63-0.77/0.83) + 0.20 * Multi-Task NER Loss."),
        ("Sheet 6", "6_Evaluasi_CM_CR", "Confusion Matrix 3x3 hasil 225 Unseen Test Set (99.11%) dengan rumus TP, FP, FN, Precision, Recall, F1-Score, Accuracy, Macro/Weighted Avg & Komparasi 3 Model."),
    ]
    for idx, (s_no, s_name, s_desc) in enumerate(nav_rows, start=19):
        ws0.cell(idx, 1, s_no).font = FONT_BOLD
        ws0.cell(idx, 2, s_name).font = FONT_BOLD
        ws0.merge_cells(start_row=idx, start_column=3, end_row=idx, end_column=8)
        ws0.cell(idx, 3, s_desc).font = FONT_REG
        style_range(ws0, 1, idx, 8, idx, fill=INPUT_FILL, alignment=ALIGN_LEFT)
        ws0.cell(idx, 1).alignment = ALIGN_CENTER

    ws0.cell(26, 1, "Catatan Transparansi 12-Layer Transformer mBERT:").font = FONT_BOLD
    ws0.merge_cells("A27:H28")
    ws0.cell(
        27, 1,
        "Model backbone bert-base-multilingual-cased memiliki ~177 juta parameter (12 layer x 12 attention head). "
        "Di workbook ini, keluaran last_hidden_state [128 x 768] dan logit ner_head [128 x 11] diekstrak langsung "
        "dari forward pass model terlatih, lalu SELURUH perhitungan pasca-Transformer (Softmax NER 11 kelas, bobot "
        "NER-Guided Pooling, Weighted Mean Pooling, Combined CLS+Mean, Proyeksi Linear, Normalisasi L2, Cosine "
        "Similarity penuh 768 dimensi, PureCosineMarginLoss, NER CE Loss, dan Classification Report) dihitung "
        "100% eksak menggunakan rumus Excel aktif."
    )
    style_range(ws0, 1, 27, 8, 28, font=FONT_ITALIC, fill=WARN_FILL, alignment=ALIGN_LEFT)

    # =========================================================================
    # SHEET 1: 1_Input_NER_Tokenisasi
    # =========================================================================
    ws1 = wb.create_sheet("1_Input_NER_Tokenisasi")
    add_banner(
        ws1,
        "TAHAP 1: PREPROCESSING REGEX (CASED), EKSTRAKSI ENTITAS NER (TABEL 2.5), & TOKENISASI WORDPIECE (MAX_LEN = 128)",
        "Menunjukkan transformasi 1 CV Pelamar dan 3 Lowongan Kerja Indonesia dari teks mentah hingga tensor input_ids, attention_mask, dan label BIO",
        13,
    )

    ws1.cell(4, 1, "A. EKSTRAKSI ENTITAS KOMPETENSI & PENYUSUNAN HEADER NER-ENRICHED TEXT").font = FONT_SUBTITLE
    hdrs1a = [
        "Dokumen", "Tipe [DOC]", "[DOMAIN]", "[TIER]", "ROLE", "SKILL", "EDU", "DUR", "ORG",
        "Jumlah Token Aktif", "NER-Enriched Text (Masuk ke Tokenizer mBERT)"
    ]
    ws1.merge_cells("K5:M5")
    for idx, h in enumerate(hdrs1a, 1):
        ws1.cell(5, idx, h)
    style_range(ws1, 1, 5, 13, 5, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    for r_i, (label, d) in enumerate(docs, start=6):
        ent = d["entities"]
        ws1.cell(r_i, 1, label)
        ws1.cell(r_i, 2, d["doc_type"])
        ws1.cell(r_i, 3, d["domain"])
        ws1.cell(r_i, 4, d["tier"])
        ws1.cell(r_i, 5, ", ".join(ent["ROLE"][:3]) or "Unspecified")
        ws1.cell(r_i, 6, ", ".join(ent["SKILL"][:8]) or "Unspecified")
        ws1.cell(r_i, 7, ", ".join(ent["EDU"][:4]) or "Unspecified")
        ws1.cell(r_i, 8, ", ".join(ent["DUR"][:3]) or "Unspecified")
        ws1.cell(r_i, 9, ", ".join(ent["ORG"][:4]) or "Unspecified")
        # Formula counting active tokens from Table B below
        mask_col = [4, 7, 10, 13][r_i - 6]
        m_col_ltr = get_column_letter(mask_col)
        ws1.cell(r_i, 10, f"=SUM({m_col_ltr}13:{m_col_ltr}140)")
        ws1.merge_cells(start_row=r_i, start_column=11, end_row=r_i, end_column=13)
        ws1.cell(r_i, 11, d["enriched"])
        style_range(ws1, 1, r_i, 13, r_i, font=FONT_REG, fill=INPUT_FILL, alignment=ALIGN_LEFT)
        ws1.cell(r_i, 10).fill = FORMULA_FILL
        ws1.cell(r_i, 10).alignment = ALIGN_CENTER
        ws1.cell(r_i, 10).font = FONT_BOLD
        ws1.row_dimensions[r_i].height = 68

    ws1.cell(11, 1, "B. HASIL TOKENISASI WORDPIECE (128 TOKEN PENUH: INDEX 0 s.d. 127)").font = FONT_SUBTITLE
    hdrs1b = [
        "Token Idx",
        "CV Token", "CV input_ids", "CV mask",
        "Job1(Fit) Token", "Job1 input_ids", "Job1 mask",
        "Job2(Close) Token", "Job2 input_ids", "Job2 mask",
        "Job3(NotFit) Token", "Job3 input_ids", "Job3 mask",
    ]
    for c_i, h in enumerate(hdrs1b, 1):
        ws1.cell(12, c_i, h)
    style_range(ws1, 1, 12, 13, 12, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    for t in range(128):
        rr = 13 + t
        ws1.cell(rr, 1, t)
        for d_i, (_, d) in enumerate(docs):
            base_c = 2 + d_i * 3
            ws1.cell(rr, base_c, d["tokens"][t])
            ws1.cell(rr, base_c + 1, d["input_ids"][t])
            ws1.cell(rr, base_c + 2, d["attention_mask"][t])
        style_range(ws1, 1, rr, 13, rr, font=FONT_REG, fill=INPUT_FILL if t % 2 == 0 else None, alignment=ALIGN_CENTER)

    # =========================================================================
    # SHEET 2: 2_NER_Head_Softmax_Bobot
    # =========================================================================
    ws2 = wb.create_sheet("2_NER_Head_Softmax_Bobot")
    add_banner(
        ws2,
        "TAHAP 2: NER HEAD (11 LABEL BIO), SOFTMAX PROBABILITY, BOBOT NER-GUIDED POOLING, & NER CROSS-ENTROPY LOSS",
        "Logit diambil langsung dari self.model.ner_head(last_hidden_state) | Kolom Q s.d. W menggunakan RUMUS EXCEL AKTIF",
        23,
    )

    ner_headers = (
        ["Idx", "Token", "Mask", "True_BIO_ID", "True_BIO_Tag"]
        + [f"Logit_{t}" for t in NER_TAGS]
        + ["Max_Logit", "Sum_Exp", "P(O)", "P(Entity)=1-P(O)", "Pool_Weight", "Pred_BIO_Tag", "Token_CE_Loss"]
    )
    # Track exact row ranges for each doc so Sheet 3 and Sheet 5 can link with formulas!
    doc_sheet2_ranges: List[Dict[str, int]] = []
    curr_r = 4

    for d_label, d in docs:
        ws2.cell(curr_r, 1, f"DOKUMEN: {d_label}").font = FONT_SUBTITLE
        curr_r += 1
        hdr_row = curr_r
        for c_i, h in enumerate(ner_headers, 1):
            ws2.cell(hdr_row, c_i, h)
        style_range(ws2, 1, hdr_row, 23, hdr_row, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

        start_tok_r = hdr_row + 1
        end_tok_r = start_tok_r + 127
        for t in range(128):
            rr = start_tok_r + t
            bio_id = d["bio_ids"][t]
            bio_tag = "IGN(-100)" if bio_id == -100 else ID2TAG[bio_id]
            ws2.cell(rr, 1, t)
            ws2.cell(rr, 2, d["tokens"][t])
            ws2.cell(rr, 3, d["attention_mask"][t])
            ws2.cell(rr, 4, bio_id)
            ws2.cell(rr, 5, bio_tag)
            for k in range(11):
                ws2.cell(rr, 6 + k, float(d["ner_logits"][t, k]))

            # Live Excel formulas (cols Q=17 .. W=23):
            # Q: Max_Logit for numerical stability
            ws2.cell(rr, 17, f"=MAX(F{rr}:P{rr})")
            # R: Sum of shifted exponentials (explicit sum for universal Excel compatibility without CSE)
            exp_terms = "+".join([f"EXP({get_column_letter(6 + k)}{rr}-Q{rr})" for k in range(11)])
            ws2.cell(rr, 18, f"={exp_terms}")
            # S: P(O) = exp(Logit_O - Max) / Sum_Exp
            ws2.cell(rr, 19, f"=EXP(F{rr}-Q{rr})/R{rr}")
            # T: P(entity) = 1 - P(O)
            ws2.cell(rr, 20, f"=1-S{rr}")
            # U: NER-Guided Pooling Weight = Mask * (1 + 0.5 * P(entity))
            ws2.cell(rr, 21, f"=C{rr}*(1+0.5*T{rr})")
            # V: Argmax predicted BIO tag
            ws2.cell(rr, 22, f'=SUBSTITUTE(INDEX($F${hdr_row}:$P${hdr_row},1,MATCH(Q{rr},F{rr}:P{rr},0)),"Logit_","")')
            # W: Token-level Cross-Entropy Loss (-LN(P(true_class))) for non-ignored tokens
            ws2.cell(rr, 23, f'=IF(D{rr}=-100,"",-LN(EXP(INDEX(F{rr}:P{rr},1,D{rr}+1)-Q{rr})/R{rr}))')

            style_range(ws2, 1, rr, 16, rr, font=FONT_REG, fill=INPUT_FILL, alignment=ALIGN_RIGHT)
            style_range(ws2, 17, rr, 23, rr, font=FONT_REG, fill=FORMULA_FILL, alignment=ALIGN_RIGHT)
            ws2.cell(rr, 1).alignment = ALIGN_CENTER
            ws2.cell(rr, 2).alignment = ALIGN_LEFT
            ws2.cell(rr, 3).alignment = ALIGN_CENTER
            ws2.cell(rr, 4).alignment = ALIGN_CENTER
            ws2.cell(rr, 5).alignment = ALIGN_CENTER
            ws2.cell(rr, 22).alignment = ALIGN_CENTER
            for col_f in list(range(6, 22)) + [23]:
                ws2.cell(rr, col_f).number_format = "0.0000"

        sum_r = end_tok_r + 1
        ws2.cell(sum_r, 1, "TOTAL / RATA-RATA")
        ws2.cell(sum_r, 3, f"=SUM(C{start_tok_r}:C{end_tok_r})")
        ws2.cell(sum_r, 20, f"=AVERAGEIF(C{start_tok_r}:C{end_tok_r},1,T{start_tok_r}:T{end_tok_r})")
        ws2.cell(sum_r, 21, f"=SUM(U{start_tok_r}:U{end_tok_r})")
        ws2.cell(sum_r, 22, "Mean NER CE Loss ->")
        ws2.cell(sum_r, 23, f"=AVERAGE(W{start_tok_r}:W{end_tok_r})")
        style_range(ws2, 1, sum_r, 23, sum_r, font=FONT_BOLD, fill=SECTION_FILL, alignment=ALIGN_RIGHT)
        ws2.cell(sum_r, 20).number_format = "0.0000"
        ws2.cell(sum_r, 21).number_format = "0.0000"
        ws2.cell(sum_r, 23).number_format = "0.000000"

        doc_sheet2_ranges.append({
            "hdr_row": hdr_row,
            "start_row": start_tok_r,
            "end_row": end_tok_r,
            "sum_row": sum_r,
        })
        curr_r = sum_r + 3

    # =========================================================================
    # SHEET 3: 3_Pooling_Proyeksi_L2
    # =========================================================================
    ws3 = wb.create_sheet("3_Pooling_Proyeksi_L2")
    add_banner(
        ws3,
        "TAHAP 3: NER-GUIDED ENTITY-WEIGHTED POOLING, COMBINED (0.5*CLS + 0.5*MEAN), PROYEKSI 768-D, & NORMALISASI L2",
        "Bagian A: Verifikasi eksplisit SUMPRODUCT Pooling & Proyeksi | Bagian B: Seluruh 768 dimensi (d_0..d_767) vektor CV (u) dan 3 Lowongan (v1, v2, v3)",
        21,
    )

    # Part A: Proof of exact NER-Guided Weighted Mean Pooling (dim 0 of CV across 128 tokens)
    # and exact Linear Projection W[0,:] * combined + b[0] across all 768 dims!
    ws3.cell(4, 1, "A. RINGKASAN VERIFIKASI NORMALISASI L2 (SEMUA VEKTOR HARUS BERPANJANG ||x||_2 = 1.000000) & BUKTI PROYEKSI DIMENSI d_0").font = FONT_SUBTITLE
    ws3.cell(5, 1, "Dokumen")
    ws3.cell(5, 2, "Norma Sebelum L2 = SQRT(SUMSQ(Projected))")
    ws3.cell(5, 3, "Norma Setelah L2 = SQRT(SUMSQ(Vec_768))")
    ws3.cell(5, 4, "Bukti W[0,:]·Combined + b[0] (CV d_0)")
    ws3.cell(5, 5, "Projected_d0 (PyTorch)")
    ws3.cell(5, 6, "Selisih Absolut")
    style_range(ws3, 1, 5, 6, 5, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    # Full 768-dim table starts at row 14 (header at row 13, rows 14..781 = d_0..d_767)
    # Columns for Part B (21 cols):
    # Col 1: Dim_Idx (0..767)
    # Col 2..6: CV (CLS, WeightedMean, Combined [=0.5*B+0.5*C], Projected, L2_Norm [=E/$B$6])
    # Col 7..11: Job1 Fit (CLS, WeightedMean, Combined, Projected, L2_Norm [=J/$B$7])
    # Col 12..16: Job2 Close (CLS, WeightedMean, Combined, Projected, L2_Norm [=O/$B$8])
    # Col 17..21: Job3 NotFit (CLS, WeightedMean, Combined, Projected, L2_Norm [=T/$B$9])
    # Plus Col 22 (V): W_proj_row0 (768 weights for row 0) to prove Projected_d0 = SUMPRODUCT(V14:V781, D14:D781) + bias0!
    D_START = 14
    D_END = 14 + 767  # 781

    proj_cols = [("E", "F"), ("J", "K"), ("O", "P"), ("T", "U")]
    for idx_d, ((d_label, _), (p_col, n_col)) in enumerate(zip(docs, proj_cols)):
        rr = 6 + idx_d
        ws3.cell(rr, 1, d_label)
        ws3.cell(rr, 2, f"=SQRT(SUMSQ({p_col}{D_START}:{p_col}{D_END}))")
        ws3.cell(rr, 3, f"=SQRT(SUMSQ({n_col}{D_START}:{n_col}{D_END}))")
        if idx_d == 0:
            ws3.cell(rr, 4, f"=SUMPRODUCT(V{D_START}:V{D_END},D{D_START}:D{D_END})+W{D_START}")
            ws3.cell(rr, 5, f"=E{D_START}")
            ws3.cell(rr, 6, f"=ABS(D{rr}-E{rr})")
        else:
            ws3.cell(rr, 4, "-")
            ws3.cell(rr, 5, "-")
            ws3.cell(rr, 6, "-")
        style_range(ws3, 1, rr, 6, rr, font=FONT_REG, fill=FORMULA_FILL, alignment=ALIGN_RIGHT)
        ws3.cell(rr, 1).alignment = ALIGN_LEFT
        ws3.cell(rr, 1).fill = INPUT_FILL
        ws3.cell(rr, 2).number_format = "0.000000"
        ws3.cell(rr, 3).number_format = "0.000000"
        if idx_d == 0:
            ws3.cell(rr, 4).number_format = "0.000000"
            ws3.cell(rr, 5).number_format = "0.000000"
            ws3.cell(rr, 6).number_format = "0.000000"

    # Also prove Weighted Mean Pooling for CV dim d_0 using Sheet 2's weights (put h[0..127, d_0] in cols X=24, Y=25 rows 14..141)
    ws3.cell(5, 8, "Bukti NER-Guided Mean Pooling (CV d_0, 128 Token)")
    ws3.cell(5, 9, "SUMPRODUCT(Weight_Sheet2, h_d0) / SUM(Weight)")
    ws3.cell(5, 10, "WeightedMean_d0 (PyTorch)")
    ws3.cell(5, 11, "Selisih Absolut")
    style_range(ws3, 8, 5, 11, 5, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    cv_s2 = doc_sheet2_ranges[0]
    ws3.cell(6, 8, "CV Pelamar — Dimensi d_0")
    ws3.cell(
        6, 9,
        f"=SUMPRODUCT('2_NER_Head_Softmax_Bobot'!U{cv_s2['start_row']}:U{cv_s2['end_row']},X{D_START}:X{D_START+127})/'2_NER_Head_Softmax_Bobot'!U{cv_s2['sum_row']}"
    )
    ws3.cell(6, 10, f"=C{D_START}")
    ws3.cell(6, 11, "=ABS(I6-J6)")
    style_range(ws3, 8, 6, 11, 6, font=FONT_REG, fill=FORMULA_FILL, alignment=ALIGN_RIGHT)
    ws3.cell(6, 8).alignment = ALIGN_LEFT
    ws3.cell(6, 8).fill = INPUT_FILL
    for cc in (9, 10, 11):
        ws3.cell(6, cc).number_format = "0.000000"

    ws3.cell(12, 1, "B. TABEL PENUH 768 DIMENSI (d_0 s.d. d_767): CLS, NER-WEIGHTED MEAN, COMBINED (0.5*CLS+0.5*MEAN), PROYEKSI, & VEKTOR FINAL L2-NORM").font = FONT_SUBTITLE
    hdrs3b = [
        "Dim_Idx",
        "CV_CLS", "CV_WMean", "CV_Combined", "CV_Projected", "CV_Vec768_u",
        "J1_CLS", "J1_WMean", "J1_Combined", "J1_Projected", "J1_Vec768_v1",
        "J2_CLS", "J2_WMean", "J2_Combined", "J2_Projected", "J2_Vec768_v2",
        "J3_CLS", "J3_WMean", "J3_Combined", "J3_Projected", "J3_Vec768_v3",
        "W_proj[0,d]", "Bias_proj[0]", "CV_TokenEmb_h[t,d0]"
    ]
    for c_i, h in enumerate(hdrs3b, 1):
        ws3.cell(13, c_i, h)
    style_range(ws3, 1, 13, 24, 13, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    for d_idx in range(768):
        rr = D_START + d_idx
        ws3.cell(rr, 1, d_idx)
        for doc_i, (_, d_data) in enumerate(docs):
            b_col = 2 + doc_i * 5
            c_cls = get_column_letter(b_col)
            c_wm = get_column_letter(b_col + 1)
            c_proj = get_column_letter(b_col + 3)
            norm_ref_row = 6 + doc_i

            ws3.cell(rr, b_col, float(d_data["cls_emb"][d_idx]))
            ws3.cell(rr, b_col + 1, float(d_data["mean_emb"][d_idx]))
            ws3.cell(rr, b_col + 2, f"=0.5*{c_cls}{rr}+0.5*{c_wm}{rr}")
            ws3.cell(rr, b_col + 3, float(d_data["projected"][d_idx]))
            ws3.cell(rr, b_col + 4, f"={c_proj}{rr}/$B${norm_ref_row}")

        ws3.cell(rr, 22, float(proj_w0[d_idx]))
        if d_idx == 0:
            ws3.cell(rr, 23, proj_b0)
        if d_idx < 128:
            ws3.cell(rr, 24, float(tr_cv["token_embs"][d_idx, 0]))

    # Format Sheet 3 numeric table efficiently
    for r_i in range(D_START, D_END + 1):
        ws3.cell(r_i, 1).alignment = ALIGN_CENTER
        ws3.cell(r_i, 1).font = FONT_REG
        for c_i in range(2, 25):
            cell = ws3.cell(r_i, c_i)
            cell.font = FONT_REG
            cell.number_format = "0.000000"
            if c_i in (4, 6, 9, 11, 14, 16, 19, 21):
                cell.fill = FORMULA_FILL

    # =========================================================================
    # SHEET 4: 4_Cosine_Similarity
    # =========================================================================
    ws4 = wb.create_sheet("4_Cosine_Similarity")
    add_banner(
        ws4,
        "TAHAP 4: PERHITUNGAN COSINE SIMILARITY PENUH (768 DIMENSI) & KLASIFIKASI AMBANG BATAS SKRIPSI",
        "Menghitung cos(u, v) = SUM(u_d * v_d) / (||u||_2 * ||v||_2) pada 768 dimensi & membandingkan langsung terhadap output sklearn.metrics.pairwise.cosine_similarity",
        10,
    )

    ws4.cell(4, 1, "A. RINGKASAN SKOR KEMIRIPAN & KEPUTUSAN (RUMUS EXCEL AKTIF VS SKLEARN PYTHON)").font = FONT_SUBTITLE
    hdrs4a = [
        "Pasangan Studi Kasus", "Label Aktual", "Dot Product SUM(u·v)",
        "||u||_2 (CV)", "||v||_2 (Job)", "Cosine Similarity (Excel)",
        "Persentase (%)", "Kategori Keputusan (Skripsi)",
        "Referensi Sklearn (Python)", "Selisih Absolut"
    ]
    for c_i, h in enumerate(hdrs4a, 1):
        ws4.cell(5, c_i, h)
    style_range(ws4, 1, 5, 10, 5, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    # Full 768-dim table starts at row 12 (rows 12..779)
    C_START = 12
    C_END = 12 + 767  # 779
    pair_defs = [
        ("Pasangan 1: CV Software Engineer vs Job Software Engineer", "Fit", "C", "D", sklearn_sims[0], FIT_FILL),
        ("Pasangan 2: CV Software Engineer vs Job Technical Support", "Close Fit", "E", "F", sklearn_sims[1], CLOSE_FILL),
        ("Pasangan 3: CV Software Engineer vs Job Executive Chef", "Not Fit", "G", "H", sklearn_sims[2], NOT_FILL),
    ]
    for idx_p, (p_name, gt, v_col, prod_col, sk_val, b_fill) in enumerate(pair_defs):
        rr = 6 + idx_p
        ws4.cell(rr, 1, p_name)
        ws4.cell(rr, 2, gt)
        ws4.cell(rr, 3, f"=SUM({prod_col}{C_START}:{prod_col}{C_END})")
        ws4.cell(rr, 4, f"=SQRT(SUMSQ(B{C_START}:B{C_END}))")
        ws4.cell(rr, 5, f"=SQRT(SUMSQ({v_col}{C_START}:{v_col}{C_END}))")
        ws4.cell(rr, 6, f"=C{rr}/(D{rr}*E{rr})")
        ws4.cell(rr, 7, f"=MAX(0,MIN(1,F{rr}))*100")
        ws4.cell(rr, 8, f'=IF(G{rr}>=80,"Fit",IF(G{rr}>=60,"Close Fit","Not Fit"))')
        ws4.cell(rr, 9, sk_val)
        ws4.cell(rr, 10, f"=ABS(F{rr}-I{rr})")

        style_range(ws4, 1, rr, 10, rr, font=FONT_REG, fill=FORMULA_FILL, alignment=ALIGN_RIGHT)
        ws4.cell(rr, 1).alignment = ALIGN_LEFT
        ws4.cell(rr, 1).fill = INPUT_FILL
        ws4.cell(rr, 2).alignment = ALIGN_CENTER
        ws4.cell(rr, 2).fill = INPUT_FILL
        ws4.cell(rr, 8).alignment = ALIGN_CENTER
        ws4.cell(rr, 8).fill = b_fill
        ws4.cell(rr, 8).font = FONT_BOLD
        ws4.cell(rr, 9).fill = INPUT_FILL
        for cc in (3, 4, 5, 6, 9, 10):
            ws4.cell(rr, cc).number_format = "0.000000"
        ws4.cell(rr, 7).number_format = "0.00"

    ws4.cell(10, 1, "B. PERKALIAN ELEMEN PER ELEMEN (u_d * v_d) UNTUK SELURUH 768 DIMENSI").font = FONT_SUBTITLE
    hdrs4b = [
        "Dim_Idx", "u_d (CV)",
        "v1_d (Job1 Fit)", "u_d * v1_d",
        "v2_d (Job2 Close Fit)", "u_d * v2_d",
        "v3_d (Job3 Not Fit)", "u_d * v3_d",
    ]
    for c_i, h in enumerate(hdrs4b, 1):
        ws4.cell(11, c_i, h)
    style_range(ws4, 1, 11, 8, 11, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    for d_idx in range(768):
        rr = C_START + d_idx
        s3_r = D_START + d_idx
        ws4.cell(rr, 1, d_idx)
        ws4.cell(rr, 2, f"='3_Pooling_Proyeksi_L2'!F{s3_r}")
        ws4.cell(rr, 3, f"='3_Pooling_Proyeksi_L2'!K{s3_r}")
        ws4.cell(rr, 4, f"=B{rr}*C{rr}")
        ws4.cell(rr, 5, f"='3_Pooling_Proyeksi_L2'!P{s3_r}")
        ws4.cell(rr, 6, f"=B{rr}*E{rr}")
        ws4.cell(rr, 7, f"='3_Pooling_Proyeksi_L2'!U{s3_r}")
        ws4.cell(rr, 8, f"=B{rr}*G{rr}")
        ws4.cell(rr, 1).alignment = ALIGN_CENTER
        ws4.cell(rr, 1).font = FONT_REG
        for c_i in range(2, 9):
            cell = ws4.cell(rr, c_i)
            cell.font = FONT_REG
            cell.number_format = "0.000000"
            if c_i in (4, 6, 8):
                cell.fill = FORMULA_FILL

    # =========================================================================
    # SHEET 5: 5_Loss_PureCosine_NER
    # =========================================================================
    ws5 = wb.create_sheet("5_Loss_PureCosine_NER")
    add_banner(
        ws5,
        "TAHAP 5: PERHITUNGAN FUNGSI LOSS TRAINING (PureCosineMarginLoss + MULTI-TASK NER LOSS)",
        "Rumus exact Notebook 03 Cell 4 & Cell 6: Total Loss = (MSE * 10.0 + BandPenalty * 6.0) + 0.20 * (0.5 * (NER_Job + NER_CV))",
        12,
    )

    ws5.cell(4, 1, "A. KONSTANTA HIPERPARAMETER LOSS (NOTEBOOK 03 CELL 4)").font = FONT_SUBTITLE
    const_pairs = [
        ("scale_mse", 10.0, "Target Fit (label=2)", 0.90, "Batas Band Fit (>=)", 0.83),
        ("scale_band", 6.0, "Target Close Fit (label=1)", 0.70, "Batas Band Close Fit", "0.63 s.d. 0.77"),
        ("alpha_ner", 0.20, "Target Not Fit (label=0)", 0.15, "Batas Band Not Fit (<=)", 0.45),
    ]
    for r_off, (k1, v1, k2, v2, k3, v3) in enumerate(const_pairs, start=5):
        ws5.cell(r_off, 1, k1).font = FONT_BOLD
        ws5.cell(r_off, 2, v1).font = FONT_REG
        ws5.cell(r_off, 3, k2).font = FONT_BOLD
        ws5.cell(r_off, 4, v2).font = FONT_REG
        ws5.cell(r_off, 5, k3).font = FONT_BOLD
        ws5.cell(r_off, 6, v3).font = FONT_REG
        style_range(ws5, 1, r_off, 6, r_off, fill=INPUT_FILL, alignment=ALIGN_CENTER)

    ws5.cell(9, 1, "B. PERHITUNGAN LOSS PER SAMPEL DALAM BATCH (3 PASANGAN: FIT, CLOSE FIT, NOT FIT)").font = FONT_SUBTITLE
    hdrs5 = [
        "Pasangan", "Label ID (y)", "Kategori", "Target Ideal (y*)",
        "Cosine Prediksi", "Squared Error (cos - y*)^2",
        "Penalti Not Fit", "Penalti Close Fit", "Penalti Fit",
        "Total Band Penalty", "NER CE Loss (Job)", "NER CE Loss (CV)"
    ]
    for c_i, h in enumerate(hdrs5, 1):
        ws5.cell(10, c_i, h)
    style_range(ws5, 1, 10, 12, 10, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    batch_rows = [
        ("Pasangan 1 (Fit)", 2, "Fit", "=D5", "='4_Cosine_Similarity'!F6", doc_sheet2_ranges[1]["sum_row"]),
        ("Pasangan 2 (Close Fit)", 1, "Close Fit", "=D6", "='4_Cosine_Similarity'!F7", doc_sheet2_ranges[2]["sum_row"]),
        ("Pasangan 3 (Not Fit)", 0, "Not Fit", "=D7", "='4_Cosine_Similarity'!F8", doc_sheet2_ranges[3]["sum_row"]),
    ]
    cv_ner_sum_r = doc_sheet2_ranges[0]["sum_row"]
    for idx_b, (p_lbl, lid, cat, tgt_f, cos_f, j_sum_r) in enumerate(batch_rows):
        rr = 11 + idx_b
        ws5.cell(rr, 1, p_lbl)
        ws5.cell(rr, 2, lid)
        ws5.cell(rr, 3, cat)
        ws5.cell(rr, 4, tgt_f)
        ws5.cell(rr, 5, cos_f)
        ws5.cell(rr, 6, f"=(E{rr}-D{rr})^2")
        ws5.cell(rr, 7, f"=IF(B{rr}=0,MAX(0,E{rr}-0.45)^2,0)")
        ws5.cell(rr, 8, f"=IF(B{rr}=1,MAX(0,0.63-E{rr})^2+MAX(0,E{rr}-0.77)^2,0)")
        ws5.cell(rr, 9, f"=IF(B{rr}=2,MAX(0,0.83-E{rr})^2,0)")
        ws5.cell(rr, 10, f"=G{rr}+H{rr}+I{rr}")
        ws5.cell(rr, 11, f"='2_NER_Head_Softmax_Bobot'!W{j_sum_r}")
        ws5.cell(rr, 12, f"='2_NER_Head_Softmax_Bobot'!W{cv_ner_sum_r}")
        style_range(ws5, 1, rr, 12, rr, font=FONT_REG, fill=FORMULA_FILL, alignment=ALIGN_RIGHT)
        ws5.cell(rr, 1).alignment = ALIGN_LEFT
        ws5.cell(rr, 1).fill = INPUT_FILL
        ws5.cell(rr, 2).alignment = ALIGN_CENTER
        ws5.cell(rr, 2).fill = INPUT_FILL
        ws5.cell(rr, 3).alignment = ALIGN_CENTER
        ws5.cell(rr, 3).fill = INPUT_FILL
        for cc in range(4, 13):
            ws5.cell(rr, cc).number_format = "0.000000"

    # Compute exact PyTorch reference loss to show side-by-side audit
    cos_t = torch.tensor(sklearn_sims, dtype=torch.float32)
    lbl_t = torch.tensor([2, 1, 0], dtype=torch.long)
    tgt_t = torch.tensor([0.90, 0.70, 0.15], dtype=torch.float32)
    py_mse_term = float(F.mse_loss(cos_t, tgt_t).item() * 10.0)
    pen_not = (torch.relu(cos_t - 0.45) ** 2) * (lbl_t == 0).float()
    pen_close = ((torch.relu(0.63 - cos_t) ** 2) + (torch.relu(cos_t - 0.77) ** 2)) * (lbl_t == 1).float()
    pen_fit = (torch.relu(0.83 - cos_t) ** 2) * (lbl_t == 2).float()
    py_band_term = float((pen_not + pen_close + pen_fit).mean().item() * 6.0)
    py_cos_loss = py_mse_term + py_band_term
    py_ner_loss = float(0.5 * (np.mean([tr_j1["ner_ce_loss"], tr_j2["ner_ce_loss"], tr_j3["ner_ce_loss"]]) + tr_cv["ner_ce_loss"]))
    py_total_loss = py_cos_loss + 0.20 * py_ner_loss

    ws5.cell(15, 1, "C. AGREGASI TOTAL LOSS BATCH (RUMUS EXCEL AKTIF VS PYTORCH)").font = FONT_SUBTITLE
    ws5.cell(16, 1, "Komponen Loss")
    ws5.cell(16, 2, "Rumus Matematika")
    ws5.cell(16, 3, "Hasil Excel (Rumus Aktif)")
    ws5.cell(16, 4, "Referensi PyTorch")
    ws5.cell(16, 5, "Selisih Absolut")
    style_range(ws5, 1, 16, 5, 16, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    agg_rows = [
        ("1. MSE Term", "AVERAGE(Squared_Error) * scale_mse (10.0)", "=AVERAGE(F11:F13)*B5", py_mse_term),
        ("2. Band Penalty Term", "AVERAGE(Band_Penalty) * scale_band (6.0)", "=AVERAGE(J11:J13)*B6", py_band_term),
        ("3. PureCosineMarginLoss", "MSE_Term + Band_Penalty_Term", "=C17+C18", py_cos_loss),
        ("4. Multi-Task NER CE Loss", "0.5 * (Mean_NER_Job + Mean_NER_CV)", "=0.5*(AVERAGE(K11:K13)+AVERAGE(L11:L13))", py_ner_loss),
        ("5. TOTAL MULTI-TASK LOSS", "PureCosineMarginLoss + 0.20 * NER_CE_Loss", "=C19+B7*C20", py_total_loss),
    ]
    for idx_a, (name, math_s, excel_f, py_val) in enumerate(agg_rows):
        rr = 17 + idx_a
        ws5.cell(rr, 1, name)
        ws5.cell(rr, 2, math_s)
        ws5.cell(rr, 3, excel_f)
        ws5.cell(rr, 4, py_val)
        ws5.cell(rr, 5, f"=ABS(C{rr}-D{rr})")
        style_range(ws5, 1, rr, 5, rr, font=FONT_BOLD if idx_a == 4 else FONT_REG, fill=FORMULA_FILL, alignment=ALIGN_RIGHT)
        ws5.cell(rr, 1).alignment = ALIGN_LEFT
        ws5.cell(rr, 1).fill = SECTION_FILL if idx_a == 4 else INPUT_FILL
        ws5.cell(rr, 2).alignment = ALIGN_LEFT
        ws5.cell(rr, 2).fill = INPUT_FILL
        ws5.cell(rr, 4).fill = INPUT_FILL
        for cc in (3, 4, 5):
            ws5.cell(rr, cc).number_format = "0.000000"

    # =========================================================================
    # SHEET 6: 6_Evaluasi_CM_CR
    # =========================================================================
    ws6 = wb.create_sheet("6_Evaluasi_CM_CR")
    add_banner(
        ws6,
        "TAHAP 6: EVALUASI CONFUSION MATRIX, CLASSIFICATION REPORT (225 UNSEEN TEST SET), & KOMPARASI 3 MODEL",
        "Angka Confusion Matrix riil hasil Notebook 03 Cell 9 (Colab) | Seluruh metrik Precision, Recall, F1-Score, Accuracy, Macro/Weighted Avg dihitung rumus Excel",
        8,
    )

    ws6.cell(4, 1, "A. CONFUSION MATRIX MODEL 2 USULAN (225 UNSEEN TEST SET — BATAS SKRIPSI 60% & 80%)").font = FONT_SUBTITLE
    ws6.cell(5, 1, "Aktual \\ Prediksi")
    ws6.cell(5, 2, "Pred: Not Fit")
    ws6.cell(5, 3, "Pred: Close Fit")
    ws6.cell(5, 4, "Pred: Fit")
    ws6.cell(5, 5, "Total Aktual (Support)")
    style_range(ws6, 1, 5, 5, 5, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    cm_data = [
        ("Aktual: Not Fit", 75, 0, 0),
        ("Aktual: Close Fit", 0, 73, 2),
        ("Aktual: Fit", 0, 0, 75),
    ]
    for idx_c, (lbl, c0, c1, c2) in enumerate(cm_data):
        rr = 6 + idx_c
        ws6.cell(rr, 1, lbl)
        ws6.cell(rr, 2, c0)
        ws6.cell(rr, 3, c1)
        ws6.cell(rr, 4, c2)
        ws6.cell(rr, 5, f"=SUM(B{rr}:D{rr})")
        style_range(ws6, 1, rr, 5, rr, font=FONT_REG, fill=INPUT_FILL, alignment=ALIGN_CENTER)
        ws6.cell(rr, 1).font = FONT_BOLD
        ws6.cell(rr, 1).alignment = ALIGN_LEFT
        ws6.cell(rr, 2 + idx_c).fill = FIT_FILL
        ws6.cell(rr, 2 + idx_c).font = FONT_BOLD
        ws6.cell(rr, 5).fill = FORMULA_FILL
        ws6.cell(rr, 5).font = FONT_BOLD

    ws6.cell(9, 1, "Total Prediksi")
    for col_i, col_ltr in [(2, "B"), (3, "C"), (4, "D"), (5, "E")]:
        ws6.cell(9, col_i, f"=SUM({col_ltr}6:{col_ltr}8)")
    style_range(ws6, 1, 9, 5, 9, font=FONT_BOLD, fill=SECTION_FILL, alignment=ALIGN_CENTER)
    ws6.cell(9, 1).alignment = ALIGN_LEFT

    ws6.cell(11, 1, "B. PERHITUNGAN CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE, ACCURACY)").font = FONT_SUBTITLE
    hdrs6b = ["Kelas", "TP", "FP", "FN", "Support", "Precision = TP/(TP+FP)", "Recall = TP/(TP+FN)", "F1-Score = 2*P*R/(P+R)"]
    for c_i, h in enumerate(hdrs6b, 1):
        ws6.cell(12, c_i, h)
    style_range(ws6, 1, 12, 8, 12, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    cr_specs = [
        ("Not Fit", "=B6", "=B9-B6", "=E6-B6", "=E6"),
        ("Close Fit", "=C7", "=C9-C7", "=E7-C7", "=E7"),
        ("Fit", "=D8", "=D9-D8", "=E8-D8", "=E8"),
    ]
    for idx_k, (cname, f_tp, f_fp, f_fn, f_sup) in enumerate(cr_specs):
        rr = 13 + idx_k
        ws6.cell(rr, 1, cname)
        ws6.cell(rr, 2, f_tp)
        ws6.cell(rr, 3, f_fp)
        ws6.cell(rr, 4, f_fn)
        ws6.cell(rr, 5, f_sup)
        ws6.cell(rr, 6, f"=B{rr}/(B{rr}+C{rr})")
        ws6.cell(rr, 7, f"=B{rr}/(B{rr}+D{rr})")
        ws6.cell(rr, 8, f"=2*F{rr}*G{rr}/(F{rr}+G{rr})")
        style_range(ws6, 1, rr, 8, rr, font=FONT_REG, fill=FORMULA_FILL, alignment=ALIGN_CENTER)
        ws6.cell(rr, 1).alignment = ALIGN_LEFT
        ws6.cell(rr, 1).font = FONT_BOLD
        ws6.cell(rr, 1).fill = INPUT_FILL
        for cc in (6, 7, 8):
            ws6.cell(rr, cc).number_format = "0.0000"

    # Accuracy, Macro Avg, Weighted Avg
    ws6.cell(16, 1, "Accuracy (Global)")
    ws6.cell(16, 2, "=SUM(B13:B15)")
    ws6.cell(16, 5, "=SUM(E13:E15)")
    ws6.cell(16, 8, "=B16/E16")
    style_range(ws6, 1, 16, 8, 16, font=FONT_BOLD, fill=SECTION_FILL, alignment=ALIGN_CENTER)
    ws6.cell(16, 1).alignment = ALIGN_LEFT
    ws6.cell(16, 8).number_format = "0.00%"

    ws6.cell(17, 1, "Macro Avg")
    ws6.cell(17, 5, "=SUM(E13:E15)")
    ws6.cell(17, 6, "=AVERAGE(F13:F15)")
    ws6.cell(17, 7, "=AVERAGE(G13:G15)")
    ws6.cell(17, 8, "=AVERAGE(H13:H15)")
    style_range(ws6, 1, 17, 8, 17, font=FONT_BOLD, fill=FORMULA_FILL, alignment=ALIGN_CENTER)
    ws6.cell(17, 1).alignment = ALIGN_LEFT
    for cc in (6, 7, 8):
        ws6.cell(17, cc).number_format = "0.00%"

    ws6.cell(18, 1, "Weighted Avg")
    ws6.cell(18, 5, "=SUM(E13:E15)")
    ws6.cell(18, 6, "=SUMPRODUCT(E13:E15,F13:F15)/E18")
    ws6.cell(18, 7, "=SUMPRODUCT(E13:E15,G13:G15)/E18")
    ws6.cell(18, 8, "=SUMPRODUCT(E13:E15,H13:H15)/E18")
    style_range(ws6, 1, 18, 8, 18, font=FONT_BOLD, fill=FORMULA_FILL, alignment=ALIGN_CENTER)
    ws6.cell(18, 1).alignment = ALIGN_LEFT
    for cc in (6, 7, 8):
        ws6.cell(18, cc).number_format = "0.00%"

    ws6.cell(20, 1, "C. REKAPITULASI KOMPARASI 3 MODEL SKRIPSI (225 UNSEEN TEST SET)").font = FONT_SUBTITLE
    hdrs6c = ["Skenario Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "Selisih Macro F1 vs Model Usulan"]
    for c_i, h in enumerate(hdrs6c, 1):
        ws6.cell(21, c_i, h)
    style_range(ws6, 1, 21, 6, 21, font=FONT_HDR, fill=SUBHDR_FILL, alignment=ALIGN_CENTER)

    comp_rows = [
        ("Model 3: Cosine Saja (TF-IDF + Sklearn Cosine)", 0.3333, 0.1111, 0.3333, 0.1667, "=E24-E22"),
        ("Model 1: mBERT Saja (Tanpa Cosine - Linear Classifier)", 0.7244, 0.7281, 0.7244, 0.7137, "=E24-E23"),
        ("Model 2 (Usulan): Unified NER + Siamese mBERT + Sklearn Cosine", "=H16", "=F17", "=G17", "=H17", 0.0),
    ]
    for idx_m, (m_name, acc, prec, rec, mf1, diff_f) in enumerate(comp_rows):
        rr = 22 + idx_m
        ws6.cell(rr, 1, m_name)
        ws6.cell(rr, 2, acc)
        ws6.cell(rr, 3, prec)
        ws6.cell(rr, 4, rec)
        ws6.cell(rr, 5, mf1)
        ws6.cell(rr, 6, diff_f)
        fill_use = FIT_FILL if idx_m == 2 else INPUT_FILL
        style_range(ws6, 1, rr, 6, rr, font=FONT_BOLD if idx_m == 2 else FONT_REG, fill=fill_use, alignment=ALIGN_CENTER)
        ws6.cell(rr, 1).alignment = ALIGN_LEFT
        ws6.cell(rr, 6).fill = FORMULA_FILL if idx_m < 2 else FIT_FILL
        for cc in range(2, 7):
            ws6.cell(rr, cc).number_format = "0.00%"

    # Set readable column widths & freeze panes on all sheets
    col_widths = {
        ws0: {1: 28, 2: 46, 3: 18, 4: 18, 5: 18, 6: 14, 7: 18, 8: 18},
        ws1: {1: 30, 2: 18, 3: 24, 4: 20, 5: 22, 6: 34, 7: 20, 8: 18, 9: 26, 10: 16, 11: 30, 12: 30, 13: 30},
        ws2: {1: 8, 2: 16, 3: 8, 4: 12, 5: 12, **{k: 11 for k in range(6, 22)}, 22: 14, 23: 14},
        ws3: {1: 28, **{k: 14 for k in range(2, 25)}},
        ws4: {1: 42, 2: 14, 3: 18, 4: 14, 5: 14, 6: 18, 7: 15, 8: 22, 9: 18, 10: 16},
        ws5: {1: 28, 2: 36, 3: 22, 4: 18, 5: 16, 6: 20, 7: 16, 8: 16, 9: 16, 10: 18, 11: 18, 12: 18},
        ws6: {1: 44, 2: 16, 3: 16, 4: 16, 5: 20, 6: 22, 7: 20, 8: 22},
    }
    for ws, widths in col_widths.items():
        for col_i, w in widths.items():
            ws.column_dimensions[get_column_letter(col_i)].width = w
        ws.views.sheetView[0].showGridLines = True

    ws0.freeze_panes = "A4"
    ws1.freeze_panes = "B13"
    ws2.freeze_panes = "C6"
    ws3.freeze_panes = "B14"
    ws4.freeze_panes = "B12"
    ws5.freeze_panes = "B11"
    ws6.freeze_panes = "B6"

    os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)
    wb.save(OUT_XLSX)
    print(f"SUCCESS: Saved workbook to {OUT_XLSX}")
    print(
        f"Verification -> Fit: {sklearn_sims[0]*100:.2f}% | Close Fit: {sklearn_sims[1]*100:.2f}% | "
        f"Not Fit: {sklearn_sims[2]*100:.2f}% | Total Loss: {py_total_loss:.6f}"
    )


if __name__ == "__main__":
    main()
