# Esperimento 3 - Sorgente del pretraining (Human vs Bridge)

**Obiettivo:** capire quanto conta il tipo di video usato nel pretraining delle latent action.
Partiamo dai modelli pre-addestrati rilasciati dagli autori e li fine-tuniamo noi su LIBERO-Goal
con la stessa ricetta del paper, poi li valutiamo (Tabella I, righe "Human" e "Bridge").

**Setup**
- Modelli di partenza: `univla-7b-human-pt` (solo video umani, Ego4D) e `univla-7b-bridge-pt` (solo robot, Bridge)
- Fine-tuning su Franklin: `vla-scripts/finetune_libero.py`, `libero_goal_no_noops`, 30000 passi,
  LoRA r32, 4 A100 80GB x batch 8 x grad_acc 4 = batch globale 128 (paper: 8 GPU x 8 x 2 = 128)
  - Script: `my_experiments/exp3_pretraining_source/submit_exp3.sh full`
  - Durata: ~4.5 s/passo, ~37 ore per modello
- Valutazione su Franklin (1 A100): `my_experiments/franklin/submit_eval.sh full <run>`,
  bf16, 10 task x 50 prove = 500 episodi, seed 7, ultimo checkpoint (passo 30000)
- Date: 24-27/11/2026

## Risultati (success rate % su LIBERO-Goal)

| Pretraining | Paper | Noi | Differenza |
|-------------|-------|-----|------------|
| Human (solo video umani) | 90.2 | **88.6** | -1.6 |
| Bridge (solo robot)      | 91.9 | **91.8** | -0.1 |
| Full (riferimento, checkpoint rilasciato, Exp 0) | 95.6 | 92.6 | -3.0 |

Log: `/work/giannotti/univla/eval_logs/<run>/libero_goal/EVAL-*.txt` (video in `rollouts/`).
Modelli: `/work/giannotti/univla/runs/<run>/`.

## Conclusioni
- **Entrambi i numeri del paper sono riprodotti** (dentro l'oscillazione di circa +-2 punti),
  e questa volta facendo noi il fine-tuning: la ricetta di addestramento rilasciata funziona.
- Bridge > Human di circa 3 punti, come nel paper: i video di un robot aiutano un po' di piu'
  dei video umani, ma **anche solo video umani portano a ~89%**, molto sopra OpenVLA (76.5 nel paper).
- Attenzione al confronto con Full: il 92.6 viene dal checkpoint fine-tunato **dagli autori** (Exp 0),
  non da un nostro fine-tuning. Per un confronto pulito servirebbe fine-tunare noi anche `univla-7b`
  (Full) su Goal al 100% con la stessa pipeline.
- Manca ancora il riferimento piu' importante: lo stesso fine-tuning **senza** pretraining sulle
  latent action (proposta Exp 5). Senza quello non sappiamo quanto dei ~90 punti viene dal pretraining.
