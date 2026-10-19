
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=FutureWarning)

NAMA_FILE = "test_mod.csv"

FITUR_KATEGORIKAL = ["color_type"]
FITUR_NUMERIK = ["length(m)", "height(cm)", "X1", "X2"]
FITUR = FITUR_KATEGORIKAL + FITUR_NUMERIK

LABEL_KELAS = {
    0: "Kelas 0",
    1: "Kelas 1",
    2: "Kelas 2"
}


def buat_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def buat_preprocessing(skalakan=True):

    langkah_numerik = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if skalakan:
        langkah_numerik.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer(
        transformers=[
            (
                "numerik",
                Pipeline(langkah_numerik),
                FITUR_NUMERIK
            ),
            (
                "kategorikal",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    ("encoder", buat_encoder())
                ]),
                FITUR_KATEGORIKAL
            )
        ],
        remainder="drop"
    )


def buat_model(pilihan):

    if pilihan == "1":
        nama = "KNN"

        model = Pipeline([
            (
                "preprocessing",
                buat_preprocessing(skalakan=True)
            ),
            (
                "classifier",
                KNeighborsClassifier(n_neighbors=5)
            )
        ])

    elif pilihan == "2":
        nama = "Pohon Keputusan berbasis Entropy"

        model = Pipeline([
            (
                "preprocessing",
                buat_preprocessing(skalakan=False)
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="entropy",
                    random_state=42
                )
            )
        ])

    elif pilihan == "3":
        nama = "Gaussian Naive Bayes"

        model = Pipeline([
            (
                "preprocessing",
                buat_preprocessing(skalakan=True)
            ),
            (
                "classifier",
                GaussianNB()
            )
        ])

    else:
        raise ValueError("Pilihan algoritma tidak tersedia.")

    return nama, model


def evaluasi_model(nama, model, X_train, X_test,
                   y_train, y_test):

    print("\n" + "=" * 60)
    print("PROSES PELATIHAN:", nama)
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n===== HASIL PERHITUNGAN =====")
    print(f"Accuracy        : {accuracy:.4f} ({accuracy:.2%})")
    print(f"Precision Macro : {precision:.4f} ({precision:.2%})")
    print(f"Recall Macro    : {recall:.4f} ({recall:.2%})")
    print(f"F1-score Macro  : {f1:.4f} ({f1:.2%})")

    print("\n===== CLASSIFICATION REPORT =====")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "Kelas 0",
            "Kelas 1",
            "Kelas 2"
        ],
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2]
    )

    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Kelas 0",
            "Kelas 1",
            "Kelas 2"
        ]
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix - " + nama)
    plt.tight_layout()

    nama_grafik = (
        "confusion_matrix_"
        + nama.lower().replace(" ", "_")
        + ".png"
    )

    plt.savefig(nama_grafik, dpi=200)
    plt.show()

    return {
        "nama": nama,
        "model": model,
        "prediksi": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():

    nama_file = NAMA_FILE

    if not os.path.exists(nama_file):
        if os.path.exists("test_mod.csv"):
            nama_file = "test_mod.csv"
        else:
            raise FileNotFoundError(
                "Dataset tidak ditemukan. Pastikan file "
                "test_mod.csv berada di folder program."
            )

    df = pd.read_csv(nama_file)

    print("=" * 60)
    print("SISTEM KLASIFIKASI MULTIKELAS")
    print("KNN, C4.5, DAN NAIVE BAYES")
    print("=" * 60)

    print("Nama dataset:", nama_file)
    print("Jumlah data:", len(df))
    print("Jumlah kolom:", len(df.columns))

    if "condition" not in df.columns:
        raise ValueError(
            "Kolom target condition tidak ditemukan."
        )

    kolom_wajib = FITUR + ["condition"]

    for kolom in kolom_wajib:
        if kolom not in df.columns:
            raise ValueError(
                f"Kolom {kolom} tidak ditemukan dalam dataset."
            )

    df["condition"] = pd.to_numeric(
        df["condition"],
        errors="coerce"
    )

    jumlah_awal = len(df)

    df = df[
        df["condition"].isin([0, 1, 2])
    ].copy()

    df["condition"] = df["condition"].astype(int)

    print("\nJumlah data awal:", jumlah_awal)
    print("Jumlah data digunakan:", len(df))
    print(
        "Jumlah data dengan label -1/kosong yang dikeluarkan:",
        jumlah_awal - len(df)
    )

    print("\nDistribusi kelas:")
    print(df["condition"].value_counts().sort_index())

    X = df[FITUR].copy()
    y = df["condition"].copy()

    for kolom in FITUR_NUMERIK:
        X[kolom] = pd.to_numeric(
            X[kolom],
            errors="coerce"
        )

    for kolom in FITUR_KATEGORIKAL:
        X[kolom] = X[kolom].astype(object)
        X[kolom] = X[kolom].where(
            X[kolom].notna(),
            np.nan
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n===== PEMBAGIAN DATA =====")
    print("Jumlah data latih:", len(X_train))
    print("Jumlah data uji:", len(X_test))

    print("\nPILIH ALGORITMA")
    print("1. K-Nearest Neighbor (KNN)")
    print("2. C4.5 / Pohon Keputusan")
    print("3. Naive Bayes")
    print("4. Jalankan ketiga algoritma")

    while True:
        pilihan = input(
            "\nMasukkan pilihan algoritma (1/2/3/4): "
        ).strip()

        if pilihan in ["1", "2", "3", "4"]:
            break

        print("Pilihan tidak valid. Masukkan 1, 2, 3, atau 4.")

    if pilihan == "4":
        daftar_pilihan = ["1", "2", "3"]
    else:
        daftar_pilihan = [pilihan]

    semua_hasil = []

    for kode in daftar_pilihan:

        nama, model = buat_model(kode)

        hasil = evaluasi_model(
            nama,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        semua_hasil.append(hasil)

    tabel_hasil = pd.DataFrame([
        {
            "Algoritma": hasil["nama"],
            "Accuracy": hasil["accuracy"],
            "Precision_Macro": hasil["precision"],
            "Recall_Macro": hasil["recall"],
            "F1_Macro": hasil["f1"]
        }
        for hasil in semua_hasil
    ])

    print("\n===== RINGKASAN EVALUASI =====")

    print(tabel_hasil.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    nama_csv = (
        "perbandingan_model_multikelas.csv"
        if len(semua_hasil) > 1
        else "hasil_evaluasi_model.csv"
    )

    tabel_hasil.to_csv(nama_csv, index=False)

    print("\nTabel evaluasi disimpan ke:", nama_csv)

    if len(semua_hasil) > 1:

        tabel_plot = tabel_hasil.set_index("Algoritma")

        tabel_plot[
            [
                "Accuracy",
                "Precision_Macro",
                "Recall_Macro",
                "F1_Macro"
            ]
        ].plot(
            kind="bar",
            figsize=(10, 6)
        )

        plt.title("Perbandingan KNN, C4.5, dan Naive Bayes")
        plt.xlabel("Algoritma")
        plt.ylabel("Skor")
        plt.ylim(0, 1)
        plt.xticks(rotation=0)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            "grafik_perbandingan_model.png",
            dpi=200
        )
        plt.show()

    print("\n===== PREDIKSI DATA BARU =====")
    print("Masukkan nilai atribut hewan.")
    print("Tekan Enter jika nilai tidak diketahui.")
    print("Ketik 'selesai' pada input pertama untuk berhenti.")

    while True:

        data_baru = {}
        batal = False
        selesai = False

        for indeks, nama_fitur in enumerate(FITUR):

            nilai = input(
                f"Masukkan {nama_fitur}: "
            ).strip()

            if indeks == 0 and nilai.lower() == "selesai":
                selesai = True
                break

            if nilai == "":
                data_baru[nama_fitur] = np.nan

            elif nama_fitur in FITUR_NUMERIK:
                try:
                    data_baru[nama_fitur] = float(nilai)

                except ValueError:
                    print("Atribut numerik harus berupa angka.")
                    print("Silakan masukkan data dari awal.\n")
                    batal = True
                    break

            else:
                data_baru[nama_fitur] = nilai

        if selesai:
            print("Program selesai.")
            break

        if batal:
            continue

        input_baru = pd.DataFrame(
            [data_baru],
            columns=FITUR
        )

        hasil_simpan = input_baru.copy()

        print("\n===== HASIL PREDIKSI =====")

        for hasil in semua_hasil:

            model = hasil["model"]
            nama = hasil["nama"]

            prediksi_baru = int(
                model.predict(input_baru)[0]
            )

            label = LABEL_KELAS.get(
                prediksi_baru,
                str(prediksi_baru)
            )

            print(f"{nama}: {label} ({prediksi_baru})")

            kolom_prediksi = (
                "Prediksi_"
                + nama.replace(" ", "_")
                .replace("/", "_")
            )

            hasil_simpan[kolom_prediksi] = prediksi_baru

        nama_file_baru = "prediksi_data_hewan_baru.csv"

        hasil_simpan.to_csv(
            nama_file_baru,
            mode="a",
            header=not os.path.exists(nama_file_baru),
            index=False
        )

        print(
            "Hasil prediksi disimpan ke:",
            nama_file_baru
        )
        print()


if __name__ == "__main__":
    main()
