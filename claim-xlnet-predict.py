from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = XLNetTokenizer.from_pretrained('./model')
model = XLNetForSequenceClassification.from_pretrained('./model')
model.to(device)

model.eval()


def predict(content):
    inputs = tokenizer(content,

                       padding='max_length',
                       truncation=True, return_tensors="pt")
    # move to gpu
    ids = inputs["input_ids"].to(device)
    idt = inputs["token_type_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, token_type_ids=idt, attention_mask=mask)
    logits = outputs[0]
    x = F.softmax(logits, dim=-1)
    active_logits = logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits,
                                         axis=1)
    return x.cpu().detach().numpy()[0][1], flattened_predictions.cpu().numpy()[0]


app = Flask(__name__)  ####
CORS(app)


@app.route("/predict", methods=['GET', 'POST'])
def run():
    if request.method == 'POST':  ####POST
        # data_fz = json.loads(request.get_data().decode('utf-8')) ####get data
        data_fz = request.get_json()
        print(data_fz)

        if data_fz is not None:
            # data_fz = request.to_dict()
            content = data_fz['content']

        else:
            return jsonify({'Bj': -1, 'Mess': '', 'type': 'Error'})  ####return -1 if no data
    else:
        return jsonify({'Bj': -2, 'Mess': '', 'type': 'Error'})  #### return -2 if not right format

    prob, label = predict(content)
    if label == 1:
        label = 'CLAIM'
    else:
        label = 'NOCLAIM'

    return jsonify({'Bj': 1, 'Mess': '', 'result': label, 'score': float(prob)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8008, debug=False)
