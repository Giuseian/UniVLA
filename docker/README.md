# Docker per UniVLA

## Idea
- **Immagine** (`Dockerfile`): solo l'ambiente (CUDA 12.1, Python 3.10, torch 2.2, dipendenze, LIBERO).
- **Codice e dati** restano sulla macchina e vengono montati nel container:

| Sulla macchina      | Nel container       |
|---------------------|---------------------|
| `~/projects/univla` | `/workspace/univla` |
| `~/data/univla`     | `/data`             |

Se il container viene cancellato, codice e dati restano intatti.
**Regola:** ogni nuova dipendenza va aggiunta a `requirements.txt` o al `Dockerfile`.

## Comandi (dalla root della repo, sulla macchina host)

```bash
# 1. Costruire l'immagine (una volta, o dopo aver modificato Dockerfile/requirements)
docker build -t univla:cu121 -f docker/Dockerfile .

# 2. Entrare nel container (lo crea al primo avvio, poi lo riusa)
bash docker/launch_container.sh

# 3. Uscire: `exit` (il container si ferma ma non viene cancellato)

# Ricreare il container da zero (es. dopo aver ricostruito l'immagine)
docker rm univla && bash docker/launch_container.sh
```

## Dipendenze: `requirements.txt` e `requirements.lock.txt`
- `requirements.txt`: la lista "umana" (cosa vogliamo). È quella da modificare.
- `requirements.lock.txt`: le versioni **esatte** di tutti i pacchetti, generate dalla lista umana. È quella che l'immagine installa, così la build è sempre identica.

Dopo aver modificato `requirements.txt`, rigenerare il lock (serve [uv](https://docs.astral.sh/uv/)):
```bash
printf "torch==2.2.0\ntorchvision==0.17.0\n" > /tmp/c.txt
uv pip compile docker/requirements.txt -c /tmp/c.txt --python-version 3.10 \
    --python-platform x86_64-manylinux_2_28 \
    --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match \
    --no-header -o docker/requirements.lock.txt
```

## Variabili già impostate nell'immagine
- `HF_HOME=/data/cache/hf`, `TORCH_HOME=/data/cache/torch`: le cache stanno nei dati, non nel container
- `MUJOCO_GL=egl`: rendering di LIBERO senza schermo
- `PYTHONPATH=/workspace/univla`: il codice della repo è importabile senza installarlo
