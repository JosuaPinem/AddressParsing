from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset
import re, os, torch, requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
id2label = model.config.id2label


OVERRIDE = {
    "Kec.": "B-KECAMATAN",
    "Kecamatan": "B-KECAMATAN",
    "Kota": "B-KOTA",
    "Kabupaten": "B-KOTA",
    "RT": "B-RT",
    "RW": "B-RW",
    "Kel.": "B-KELURAHAN",
    "Kelurahan": "B-KELURAHAN"
}

def tokenize_address(text: str):
    text = re.sub(r'([.,/])', r' \1 ', text)
    tokens = text.split()
    return tokens

def postprocess(preds):
    fixed = []
    seen_address = set()
    for tok, lab in preds:
        if tok in OVERRIDE:
            lab = OVERRIDE[tok]

        if tok in seen_address:
            continue

        fixed.append((tok, lab))
        seen_address.add(tok)

    return fixed

def predict_address(text: str):
    tokens = tokenize_address(text)
    encoding = tokenizer(
        tokens,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    word_ids = encoding.word_ids(batch_index=0)
    preds = []
    seen = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id in seen:
            continue
        seen.add(word_id)

        label_id = predictions[0][idx].item()
        label = id2label[label_id]
        preds.append((tokens[word_id], label))

    preds = postprocess(preds)

    return preds

def extract_entities(preds):
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

    skip_keywords = ["RT", "RW"]

    current_entity = None

    for tok, lab in preds:
        if tok in OVERRIDE:
            lab = OVERRIDE[tok]

        if tok in skip_keywords:
            continue
        
        if lab.startswith("B-"):
            ent_type = lab.split("-", 1)[1]
            current_entity = ent_type

            if ent_type == "JALAN":
                entities["Jalan"] = tok
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

            if ent_type == "JALAN":
                entities["Jalan"] += " " + tok
            elif ent_type == "KELURAHAN" and tok not in skip_keywords:
                entities["Kelurahan"] += " " + tok
            elif ent_type == "KECAMATAN" and tok not in skip_keywords:
                entities["Kecamatan"] += " " + tok
            elif ent_type == "KOTA" and tok not in skip_keywords:
                entities["Kota/Kabupaten"] += " " + tok
            elif ent_type == "PROVINSI":
                entities["Provinsi"] += " " + tok
            elif ent_type == "KODEPOS":
                entities["Kode Pos"] += tok
            elif ent_type == "RT":
                entities["RT"] += " " + tok
            elif ent_type == "RW":
                entities["RW"] += " " + tok

        else:
            current_entity = None

    for k in entities:
        entities[k] = entities[k].strip()

    if not entities["Kelurahan"]:
        entities["Kelurahan"] = None
    if not entities["Kecamatan"]:
        entities["Kecamatan"] = None

    return entities

def reformatingKodePos(entities):
    url = "https://kodepos.posindonesia.co.id/CariKodepos"

    informasi = entities.get("Kode Pos", "")

    data = {"kodepos": informasi}
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', id='list-data')
        
        if table:
            rows = table.find_all('tr')[1:]
            for row in rows:
                columns = row.find_all('td')
                
                if len(columns) > 0:  # Cek jika kolom ada
                    kelurahan_data = columns[2].get_text(strip=True)
                    kecamatan_data = columns[3].get_text(strip=True)
                    kota_data = columns[4].get_text(strip=True)
                    provinsi_data = columns[5].get_text(strip=True)

                    if entities["Kota/Kabupaten"] == kota_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities  

                    elif entities["Kecamatan"] == kecamatan_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities

                    elif entities["Kelurahan"] == kelurahan_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities

                    elif entities["Kecamatan"] == kelurahan_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities 

                    elif entities["Kelurahan"] == kecamatan_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities

            print("Tidak ada kecocokan data.")
            return entities
        else:
            print("Tabel tidak ditemukan.")
            return entities

    else:
        print(f"Error: Status Code {response.status_code}")
        return entities

def reformatingNonKodePos(entities):
    url = "https://kodepos.posindonesia.co.id/CariKodepos"

    informasi = entities.get("Kota/Kabupaten", "")

    data = {"kodepos": informasi}
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', id='list-data')
        
        if table:
            rows = table.find_all('tr')[1:]
            for row in rows:
                columns = row.find_all('td')
                
                if len(columns) > 0:
                    kelurahan_data = columns[2].get_text(strip=True)
                    kecamatan_data = columns[3].get_text(strip=True)
                    kota_data = columns[4].get_text(strip=True)
                    provinsi_data = columns[5].get_text(strip=True)

                    if entities["Kota/Kabupaten"] == kota_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities

                    elif entities["Kecamatan"] == kecamatan_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities

                    elif entities["Kelurahan"] == kelurahan_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities

                    elif entities["Kecamatan"] == kelurahan_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities

                    elif entities["Kelurahan"] == kecamatan_data:
                        entities["Kelurahan"] = kelurahan_data
                        entities["Kecamatan"] = kecamatan_data
                        entities["Kota/Kabupaten"] = kota_data
                        entities["Provinsi"] = provinsi_data
                        return entities

            return entities
        else:
            return entities
    else:
        return entities