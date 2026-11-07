import os
import warnings

import joblib
import numpy as np
from flask import Flask, render_template, request

# Model dilatih dengan DataFrame; input di sini berupa array, jadi peringatan nama fitur diabaikan.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = joblib.load(os.path.join(BASE_DIR, "model_harga_hp.joblib"))
model = data["model"]
fitur = data["fitur"]

KATEGORI = ["Murah", "Menengah", "Mahal", "Sangat Mahal"]

# Metadata formulir. Rentang mengikuti rentang nilai pada dataset latih.
FIELDS = [
    {"name": "ram", "label": "RAM", "unit": "MB", "min": 256, "max": 3998,
     "hint": "Contoh: 2048 untuk RAM 2 GB"},
    {"name": "battery_power", "label": "Kapasitas baterai", "unit": "mAh", "min": 501, "max": 1998,
     "hint": "Contoh: 1500"},
    {"name": "px_width", "label": "Lebar resolusi layar", "unit": "piksel", "min": 500, "max": 1998,
     "hint": "Contoh: 1080"},
    {"name": "px_height", "label": "Tinggi resolusi layar", "unit": "piksel", "min": 0, "max": 1960,
     "hint": "Contoh: 1280"},
    {"name": "int_memory", "label": "Memori internal", "unit": "GB", "min": 2, "max": 64,
     "hint": "Contoh: 32"},
]

LABEL = {f["name"]: f["label"] for f in FIELDS}
UNIT = {f["name"]: f["unit"] for f in FIELDS}

CONTOH = {
    "rendah": {"ram": 600, "battery_power": 800, "px_width": 700, "px_height": 600, "int_memory": 8},
    "sedang": {"ram": 1500, "battery_power": 1200, "px_width": 1000, "px_height": 900, "int_memory": 24},
    "tinggi": {"ram": 2600, "battery_power": 1500, "px_width": 1300, "px_height": 1300, "int_memory": 40},
    "sangat_tinggi": {"ram": 3600, "battery_power": 1800, "px_width": 1600, "px_height": 1700, "int_memory": 60},
}


def angka(x):
    """Format angka gaya Indonesia: 2217.5 -> 2.217,5 dan 1500 -> 1.500."""
    if float(x).is_integer():
        return f"{int(x):,}".replace(",", ".")
    return f"{x:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def jalur_keputusan(X_baru):
    """Ambil pertanyaan yang dilewati input ini dari akar sampai daun."""
    tree = model.tree_
    langkah = []
    for node in model.decision_path(X_baru).indices:
        if tree.children_left[node] == tree.children_right[node]:
            continue  # daun, tidak ada pertanyaan
        nama = fitur[tree.feature[node]]
        nilai = float(X_baru[0][fitur.index(nama)])
        batas = float(tree.threshold[node])
        kecil = nilai <= batas
        langkah.append({
            "fitur": LABEL.get(nama, nama),
            "unit": UNIT.get(nama, ""),
            "nilai": angka(nilai),
            "batas": angka(batas),
            "operator": "≤" if kecil else ">",
        })
    return langkah


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    error = None
    nilai = {}

    if request.method == "POST":
        mentah = {f["name"]: request.form.get(f["name"], "").strip() for f in FIELDS}
        try:
            nilai = {k: float(v) for k, v in mentah.items()}
            di_luar = [f["label"] for f in FIELDS
                       if not (f["min"] <= nilai[f["name"]] <= f["max"])]
            if di_luar:
                error = ("Nilai di luar rentang data latih pada: " + ", ".join(di_luar) +
                         ". Perbaiki sesuai rentang yang tertera di bawah kolom.")
            else:
                X_baru = np.array([[nilai[f] for f in fitur]])
                kelas = int(model.predict(X_baru)[0])
                proba = model.predict_proba(X_baru)[0]
                hasil = {
                    "kelas": kelas,
                    "nama": KATEGORI[kelas],
                    "probabilitas": [
                        {"nama": KATEGORI[int(c)], "persen": round(float(p) * 100, 1), "aktif": int(c) == kelas}
                        for c, p in zip(model.classes_, proba)
                    ],
                    "jalur": jalur_keputusan(X_baru),
                }
        except (ValueError, KeyError):
            error = "Semua kolom wajib diisi dengan angka."

    return render_template(
        "index.html",
        fields=FIELDS,
        kategori=KATEGORI,
        hasil=hasil,
        error=error,
        nilai=mentah if request.method == "POST" else {},
        contoh=CONTOH,
        kedalaman=model.get_depth(),
        jumlah_daun=model.get_n_leaves(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
