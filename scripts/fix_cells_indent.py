import json

nb_path = 'notebooks/03_optuna_tuning_siamese_mbert.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell 5
c5 = nb['cells'][5]['source']
for i, l in enumerate(c5):
    if 'logits, _, _, cos_sim = model' in l and not l.startswith('                '):
        c5[i] = '                    logits, _, _, cos_sim = model(j_ids, j_mask, c_ids, c_mask)\n'
    if 'loss = criterion(logits, cos_sim, lbl)' in l and 'v_losses' in c5[min(i+1, len(c5)-1)]:
        c5[i] = '                    v_loss = criterion(logits, cos_sim, lbl)\n'

# Fix Cell 6
c6 = nb['cells'][6]['source']
for i, l in enumerate(c6):
    if 'logits, _, _, cos_sim = best_model' in l and not l.startswith('            '):
        c6[i] = '                logits, _, _, cos_sim = best_model(j_ids, j_mask, c_ids, c_mask)\n'
    if 'loss = criterion(logits, cos_sim, lbl)' in l and 'v_losses' in c6[min(i+1, len(c6)-1)]:
        c6[i] = '                v_loss = criterion(logits, cos_sim, lbl)\n'

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Indentation fixed.")
