"""Risultati "leggeri" scaricabili direttamente dal notebook.

Su Kaggle scaricare uno zip dal pannello Output è scomodo. Qui i file piccoli (metriche, CSV per
immagine, curve; senza i pesi .pt) diventano uno zip codificato in base64 dentro un link HTML:
- in sessione si scarica con un clic;
- il contenuto resta salvato nel .ipynb, quindi basta scaricare il notebook per avere anche i risultati.
"""
import base64
import io
import zipfile
from pathlib import Path


def light_zip(root: str | Path, exclude_suffixes: tuple[str, ...] = (".pt",)) -> bytes:
    """Zip in memoria di tutti i file sotto root, tranne quelli con le estensioni escluse."""
    root = Path(root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix not in exclude_suffixes:
                z.write(f, f.relative_to(root.parent))
    return buf.getvalue()


def download_link_html(root: str | Path, name: str, exclude_suffixes: tuple[str, ...] = (".pt",)) -> str:
    data = light_zip(root, exclude_suffixes)
    b64 = base64.b64encode(data).decode()
    return (f'<a download="{name}" href="data:application/zip;base64,{b64}" '
            f'style="font-size:1.1em">⬇ Scarica {name} ({len(data) / 1024:.0f} KB)</a>')
