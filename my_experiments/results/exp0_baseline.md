# Esperimento 0 - Baseline LIBERO

**Obiettivo:** verificare che i checkpoint fine-tunati rilasciati dagli autori
(`univla-7b-224-sft-libero`) riproducano i numeri del paper (Tabella I, riga "Full").

**Setup**
- Macchina: 9950x, RTX 4090 (24 GB), container Docker `univla:cu121`
- Script: `my_experiments/exp0_baseline/run_exp0.sh all`
- bf16, 10 task x 50 prove = 500 episodi per suite, seed 7 (default degli autori)
- Date: 19-20/11/2026 (circa 13 ore in totale)

## Risultati (success rate %)

| Suite   | Paper | Noi  | Differenza |
|---------|-------|------|------------|
| Spatial | 96.5  | 96.0 | -0.5 |
| Object  | 96.8  | 95.6 | -1.2 |
| Goal    | 95.6  | 92.6 | -3.0 |
| Long    | 92.0  | 90.4 | -1.6 |
| **Media** | **95.2** | **93.7** | **-1.5** |

Log: `/data/eval_logs/exp0_baseline/<suite>/EVAL-*.txt` (video in `rollouts/`).

## Conclusioni
- Con 500 episodi l'oscillazione naturale e' di circa +-1-2 punti.
- Spatial, Object e Long: **riprodotti** (dentro l'oscillazione).
- Goal: -3 punti, **al limite**.
- Tutte le suite sono leggermente sotto il paper: possibile piccola differenza sistematica
  (paper = run migliore o media di piu' seed? hardware/librerie diverse?).
- In generale: i modelli rilasciati funzionano come dichiarato (-1.5 punti in media).

## Possibili verifiche (opzionali)
- Rilanciare Goal con un altro seed per capire se il -3 e' casuale.
