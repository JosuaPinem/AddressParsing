from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
import re, os, torch, requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
id2label = model.config.id2label


# Override mapping for special tokens
OVERRIDE = {
    "Kec.": "B-KECAMATAN",
    "Kecamatan": "B-KECAMATAN",
    "Kota": "B-KOTA",
    "Kabupaten": "B-KOTA",
    "RT": "B-RT",  # Treat RT and RW as B-RT, B-RW
    "RW": "B-RW",  # Avoid RT/RW as separate entities
    "Kel.": "B-KELURAHAN",
    "Kelurahan": "B-KELURAHAN"
}

# Tokenizer function
def tokenize_address(text: str):
    # Pisahkan koma, titik, slash agar sesuai dengan training
    text = re.sub(r'([,/])', r' \1 ', text)
    tokens = text.split()
    return tokens

# Postprocess function for overriding labels
def postprocess(preds):
    fixed = []
    seen_address = set()  # Set untuk melacak alamat yang sudah diberi label
    for tok, lab in preds:
        # Override label sesuai mapping
        if tok in OVERRIDE:
            lab = OVERRIDE[tok]
        
        # Jika alamat belum ada di set, tambahkan
        if tok not in seen_address:
            fixed.append((tok, lab))
            seen_address.add(tok)
        else:
            # Jika token sudah ada, tetap tambahkan tapi jangan skip
            fixed.append((tok, lab))
    
    return fixed


# Function for prediction
def predict_address(text: str):
    # Step 1: Tokenisasi sesuai training
    tokens = tokenize_address(text)
    encoding = tokenizer(
        tokens,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    # Step 2: Inference
    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    # Step 3: Mapping prediksi ke label
    word_ids = encoding.word_ids(batch_index=0)
    preds = []
    seen = set()  # track word indices yang sudah dipakai
    for idx, word_id in enumerate(word_ids):
        if word_id is None:  # [CLS], [SEP]
            continue
        if word_id in seen:  # Skip subword duplikat
            continue
        seen.add(word_id)

        label_id = predictions[0][idx].item()
        label = id2label[label_id]
        preds.append((tokens[word_id], label))

    # Step 4: Postprocess (override label penting)
    preds = postprocess(preds)

    return preds

def tokenize_rtrw(text: str):
    # Pisahkan koma, titik, slash agar sesuai dengan training
    text = re.sub(r'([,/.])', r' \1 ', text)
    tokens = text.split()
    skip_key = ["RT", "RW", ".", "Rt", "Rw", "rt", "rw", "rT", "rW"]
    final_rtrw = ""
    for i in tokens:
        if i in skip_key:
            continue
        final_rtrw += i + " " 

    return final_rtrw.strip()


def extract_entities(preds):
    # Initialize entity dictionary
    entities = {
        "Jalan": "",
        "Kelurahan": "",
        "Kecamatan": "",
        "Kota/Kabupaten": "",
        "Provinsi": "",
        "Kode Pos": "",
        "RT": "",
        "RW": ""
    }

    # For skipping unwanted keywords like RT and RW
    skip_keywords = ["RT", "RW", "Kec.", "kec.", "kec", "Kec", "Kel", "Kel.", "kel", "Kel"]

    current_entity = None

    for tok, lab in preds:
        # Check if token is to be overridden or skipped
        if tok in OVERRIDE:
            lab = OVERRIDE[tok]  # Override with corresponding label
        
        # Skip RT, RW tokens while still preserving their data
        if tok in skip_keywords:
            continue  # Skip if token is "RT" or "RW"
        
        if lab.startswith("B-"):
            ent_type = lab.split("-", 1)[1]
            current_entity = ent_type

            # Process entities based on B-type labels
            if ent_type == "JALAN":
                if "Jalan" not in entities or not entities["Jalan"]:
                    entities["Jalan"] = tok 
                else:
                    entities["Jalan"] += " " + tok
            elif ent_type == "KELURAHAN" and tok not in skip_keywords:
                entities["Kelurahan"] = tok
            elif ent_type == "KECAMATAN" and tok not in skip_keywords: 
                entities["Kecamatan"] = tok
            elif ent_type == "KOTA" and tok not in skip_keywords: 
                entities["Kota/Kabupaten"] = tok
            elif ent_type == "PROVINSI":
                entities["Provinsi"] = tok
            elif ent_type == "KODEPOS":
                entities["Kode Pos"] = tok
            elif ent_type == "RT":
                entities["RT"] = tok
            elif ent_type == "RW":
                entities["RW"] = tok

        elif lab.startswith("I-") and current_entity:
            ent_type = lab.split("-", 1)[1]

            # Add tokens to respective entities (ignore if skipped)
            if ent_type == "JALAN":
                entities["Jalan"] += " " + tok
            elif ent_type == "KELURAHAN" and tok not in skip_keywords:  # Skip Kelurahan
                entities["Kelurahan"] += " " + tok
            elif ent_type == "KECAMATAN" and tok not in skip_keywords:  # Skip Kecamatan
                entities["Kecamatan"] += " " + tok
            elif ent_type == "KOTA" and tok not in skip_keywords:  # Always keep Kota
                entities["Kota/Kabupaten"] += " " + tok
            elif ent_type == "PROVINSI":
                entities["Provinsi"] += " " + tok
            elif ent_type == "KODEPOS":
                entities["Kode Pos"] += tok  # kodepos biasanya angka tanpa spasi
            elif ent_type == "RT":
                entities["RT"] += " " + tok
            elif ent_type == "RW":
                entities["RW"] += " " + tok

        else:
            current_entity = None

    if entities["RT"] not in ["", None]:
        entities["RT"] = tokenize_rtrw(entities["RT"])
    if entities["RW"] not in ["", None]:
        entities["RW"] = tokenize_rtrw(entities["RW"])
    # Clean up empty or unwanted fields
    for k in entities:
        entities[k] = entities[k].strip()

    # Remove Kelurahan and Kecamatan if they are still empty
    if not entities["Kelurahan"]:
        entities["Kelurahan"] = None
    if not entities["Kecamatan"]:
        entities["Kecamatan"] = None

    return entities

def reformatingKodePos(entities):
    url = "https://kodepos.posindonesia.co.id/CariKodepos"

    informasi = entities["Kode Pos"]
    data = {"kodepos": informasi}
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', id='list-data')
        
        if table:
            kelurahan_list = []
            kecamatan_list = []
            kota_list = []
            provinsi_list = []
            alamat_list = []

            rows = table.find_all('tr')[1:]
            for row in rows:
                columns = row.find_all('td')
                if len(columns) > 0:
                    kelurahan_data = columns[2].get_text(strip=True)
                    kecamatan_data = columns[3].get_text(strip=True)
                    kota_data = columns[4].get_text(strip=True)
                    provinsi_data = columns[5].get_text(strip=True)

                    kelurahan_list.append(kelurahan_data)
                    kecamatan_list.append(kecamatan_data)
                    kota_list.append(kota_data)
                    provinsi_list.append(provinsi_data)

                    alamat_list.append(f"{kelurahan_data} {kecamatan_data} {kota_data} {provinsi_data}")

            if len(kelurahan_list) == 0 and len(kecamatan_list) == 0 and len(kota_list) == 0 and len(provinsi_list)==0:
                if entities.get("Kota/Kabupaten") not in [None, ""]:
                    reformating = reformatingNonKodePos(entities, info = "kota")
                    return reformating
                elif entities.get("Kecamatan") not in [None, ""]:
                    reformating = reformatingNonKodePos(entities, info = "kecamatan")
                    return reformating
                elif  entities.get("Kelurahan") not in [None, ""]:
                    reformating = reformatingNonKodePos(entities, info = "kelurahan")
                    return reformating
                else:
                    return entities
            # Gabungkan entities
            alamat_entities = f"{entities['Kelurahan']} {entities['Kecamatan']} {entities['Kota/Kabupaten']} {entities['Provinsi']}"

            # Fuzzy match
            best_match = process.extractOne(alamat_entities, alamat_list, scorer=fuzz.ratio)

            print(f"Alamat Entities: {alamat_entities}")
            print(f"Best Match: {best_match}")

            threshold = 70
            if best_match and best_match[1] >= threshold:
                # cari index dari best_match di alamat_list
                idx = alamat_list.index(best_match[0])

                entities["Kelurahan"] = kelurahan_list[idx]
                entities["Kecamatan"] = kecamatan_list[idx]
                entities["Kota/Kabupaten"] = kota_list[idx]
                entities["Provinsi"] = provinsi_list[idx]

            return entities
        else:
            print("Tabel tidak ditemukan.")
            return entities
    else:
        print(f"Error: Status Code {response.status_code}")
        return entities


def reformatingNonKodePos(entities, info):
    url = "https://kodepos.posindonesia.co.id/CariKodepos"
    if info == "kota":
        informasi = entities["Kota/Kabupaten"]
    elif info == "kecamatan":
        informasi = entities["Kecamatan"]
    elif info == "kelurahan":
        informasi = entities["Kelurahan"]

    data = {"kodepos": informasi}
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', id='list-data')
        
        if table:
            kelurahan_list = []
            kecamatan_list = []
            kota_list = []
            provinsi_list = []
            alamat_list = []

            rows = table.find_all('tr')[1:]
            for row in rows:
                columns = row.find_all('td')
                if len(columns) > 0:
                    kelurahan_data = columns[2].get_text(strip=True)
                    kecamatan_data = columns[3].get_text(strip=True)
                    kota_data = columns[4].get_text(strip=True)
                    provinsi_data = columns[5].get_text(strip=True)

                    kelurahan_list.append(kelurahan_data)
                    kecamatan_list.append(kecamatan_data)
                    kota_list.append(kota_data)
                    provinsi_list.append(provinsi_data)

                    alamat_list.append(f"{kelurahan_data} {kecamatan_data} {kota_data} {provinsi_data}")

            # Gabungkan entities
            alamat_entities = f"{entities['Kelurahan']} {entities['Kecamatan']} {entities['Kota/Kabupaten']} {entities['Provinsi']}"

            # Fuzzy match
            best_match = process.extractOne(alamat_entities, alamat_list, scorer=fuzz.ratio)

            print(f"Alamat Entities: {alamat_entities}")
            print(f"Best Match: {best_match}")

            threshold = 70
            if best_match and best_match[1] >= threshold:
                # cari index dari best_match di alamat_list
                idx = alamat_list.index(best_match[0])

                entities["Kelurahan"] = kelurahan_list[idx]
                entities["Kecamatan"] = kecamatan_list[idx]
                entities["Kota/Kabupaten"] = kota_list[idx]
                entities["Provinsi"] = provinsi_list[idx]

            return entities
        else:
            print("Tabel tidak ditemukan.")
            return entities
    else:
        print(f"Error: Status Code {response.status_code}")
        return entities