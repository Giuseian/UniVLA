# Esperimento 1 - Stabilita' dei codici del LAM (metrica nostra)

**Domanda:** i codici "del compito" del LAM descrivono il movimento del robot e ignorano i
cambiamenti visivi inutili (luce, colori, telecamera)? E i codici "del resto" assorbono questi
cambiamenti, come sostiene il paper?

Il paper lo mostra solo in modo qualitativo (esempi e figure): questa misura numerica e' nostra.

**Setup**
- Modello: LAM stage-2 rilasciato (`lam-stage-2.ckpt`), non modificato
- Due gruppi di codici per ogni coppia di immagini (4 token, valori 0-15):
  - codici del compito (`indices`): quelli che il robot impara a prevedere
  - codici del resto (`indices_uncontrol`): codebook dello stage 1, per i cambiamenti non legati al compito
- Dati: LIBERO RLDS, 4 suite x 200 coppie (frame t e t+11, stessa distanza del fine-tuning), seed 7
- Misura: % di token uguali a quelli della coppia originale dopo la modifica
- Script: `my_experiments/exp1_lam/run_exp1.sh full` (circa 5 minuti sulla 9950x)
- Output: `/data/eval_logs/exp1_lam/full/` (results.md, results.json, codes.npz, examples.png)

## Risultati (media delle 4 suite, % di token uguali)

| Modifica | Codici del compito | Codici del resto |
|---|---|---|
| nessuna (controllo) | 100.0 | 100.0 |
| piu' luce (entrambe) | 90.7 | 99.8 |
| meno luce (entrambe) | 95.2 | 99.8 |
| colori (entrambe) | 88.1 | 99.7 |
| sfocatura (entrambe) | 74.0 | 99.1 |
| rumore (entrambe) | 76.4 | 99.3 |
| piu' luce (solo seconda) | 91.5 | 99.9 |
| colori (solo seconda) | 87.8 | 99.8 |
| telecamera spostata 4 px (solo seconda) | 59.8 | 98.9 |
| telecamera spostata 8 px (solo seconda) | 48.7 | 97.8 |
| telecamera spostata 16 px (solo seconda) | 39.1 | 94.1 |
| **controllo:** robot fermo (t, t) | 4.0 | 24.9 |
| **controllo:** movimento successivo (t+11, t+22) | 36.6 | 97.7 |
| **controllo:** coppie a caso (livello "caso") | 27.5 | 97.7 |

## Varieta' dei codici (coppie originali)

| Suite | Codici del compito: combinazioni diverse su 200 | Codici del resto: combinazioni diverse su 200 | Codici del resto: % della combinazione piu' comune |
|---|---|---|---|
| Spatial | 124 | 7 | 85.5 |
| Object | 118 | 1 | 100.0 |
| Goal | 160 | 1 | 100.0 |
| Long | 168 | 4 | 95.0 |

## Conclusioni

1. **I codici del compito descrivono davvero il movimento.** Con un movimento diverso scendono quasi
   al livello "caso" (36.6 vs 27.5), e il robot fermo ha codici completamente diversi (4.0).
2. **Sono robusti a luce e colori** (88-95%), meno a sfocatura e rumore (74-76%).
3. **Non sono robusti allo spostamento della telecamera.** Bastano 4 pixel (meno del 2% dell'immagine)
   per cambiare il 40% dei token; con 16 pixel si e' vicini al livello "caso" (39.1 vs 27.5).
   Il LAM interpreta il movimento della telecamera come un movimento del robot.
4. **I codici del resto su LIBERO sono quasi costanti** (1-7 combinazioni su 200; in Object e Goal
   sempre la stessa). Reagiscono solo a "fermo / non fermo" (24.9). Non assorbono lo spostamento
   della telecamera (restano 94-99%), quindi la separazione "compito / resto" dichiarata nel paper
   **non si vede** in questo test.

## Limiti
- Su LIBERO la telecamera e' fissa: il LAM non ha mai visto spostamenti di telecamera in questi dati
  (li ha visti nel pre-addestramento, es. video umani). Le modifiche sono sintetiche.
- Per i risultati su LIBERO (Exp 0) non e' un problema, perche' la telecamera non si muove.
  Potrebbe esserlo su un robot reale con telecamera che vibra o si sposta.
- 200 coppie per suite, un solo seed.
