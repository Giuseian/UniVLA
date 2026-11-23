# Esperimento 2 - Senza storia (history latent actions)

**Obiettivo:** capire quanto conta la "storia" (le latent action dei passi precedenti
messe nel prompt) confrontando con il paper (Tabella V).

**Setup**
- Macchina: 9950x, RTX 4090 (24 GB), container Docker `univla:cu121`
- Script: `my_experiments/exp2_no_history/run_exp2.sh all` (`--use_history False`)
- Stessi checkpoint di Exp 0 (`univla-7b-224-sft-libero`), bf16, 500 episodi per suite, seed 7
- Solo Goal e Long (le suite della Tabella V)
- Date: 20/11/2026

**Attenzione:** i checkpoint rilasciati sono stati **addestrati con la storia**; noi la togliamo
**solo durante la valutazione**. Nel paper invece il modello senza storia e' addestrato senza storia.
Quindi il confronto con la Tabella V non e' identico.

## Risultati (success rate %)

| Suite | Paper senza storia | Paper con storia | **Noi senza storia** | Noi con storia (Exp 0) |
|-------|-------------------|------------------|----------------------|------------------------|
| Goal  | 95.0 | 95.6 | **94.2** | 92.6 |
| Long  | 88.1 | 92.0 | **90.6** | 90.4 |

Log: `/data/eval_logs/exp2_no_history/<suite>/EVAL-*.txt` (video in `rollouts/`).

## Conclusioni
- Togliere la storia **non peggiora** il modello: Goal +1.6, Long +0.2 rispetto a Exp 0.
- Con 500 episodi l'oscillazione e' di circa +-2 punti per ogni numero, quindi queste
  differenze sono **rumore**: con e senza storia vanno uguale.
- Il paper dice che la storia aiuta soprattutto su Long (+3.9 punti). **Noi non vediamo
  questo effetto** almeno quando la storia viene tolta solo in valutazione.
- Interpretazione: con questi checkpoint la policy sembra usare poco la storia; decide
  soprattutto guardando l'immagine attuale e l'istruzione.
- Per una conclusione definitiva servirebbe addestrare un modello senza storia (costoso)
  oppure ripetere con piu' seed.
