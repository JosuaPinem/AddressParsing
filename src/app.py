from flask import Flask, request, jsonify
from service.service import predict_address, reformatingKodePos, reformatingNonKodePos, extract_entities

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return jsonify({"message": "Flask berhasil di jalankan!"}, 200)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'address' not in data:
        return jsonify({"error": "Invalid input, 'address' field is required."}, 400)
    address = data['address']
    preds = extract_entities(predict_address(address))

    if preds.get("Kode Pos") not in [None, ""]:
        reformating = reformatingKodePos(preds)
    elif preds.get("Kota/Kabupaten") not in [None, ""]:
        reformating = reformatingNonKodePos(preds, info = "kota"),
    elif preds.get("Kecamatan") not in [None, ""]:
        reformating = reformatingNonKodePos(preds, info = "kecamatan")
    elif  preds.get("Kelurahan") not in [None, ""]:
        reformating = reformatingNonKodePos(preds, info = "kelurahan")
    else:
        reformating = preds

    return jsonify({"predictions": reformating}, 200)

if __name__ == '__main__':
    app.run(debug=True)