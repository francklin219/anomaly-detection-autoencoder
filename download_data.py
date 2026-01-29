import os
import zipfile
import urllib.request

ZIP_URL = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
OUT_DIR = os.path.join("data", "raw")
ZIP_PATH = os.path.join(OUT_DIR, "ECG5000.zip")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading ECG5000.zip ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print("✅ Downloaded:", ZIP_PATH)

    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(OUT_DIR)

    # Optionnel: affichage du contenu
    files = os.listdir(OUT_DIR)
    print("✅ Extracted files in data/raw:")
    for f in files:
        print(" -", f)

    # Vérif attendue
    expected = ["ECG5000_TRAIN.txt", "ECG5000_TEST.txt"]
    missing = [e for e in expected if not os.path.exists(os.path.join(OUT_DIR, e))]
    if missing:
        print("Missing expected files:", missing)
    else:
        print("Found TRAIN/TEST txt files. Ready for preprocess.")

if __name__ == "__main__":
    main()
