from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
# from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = XLNetTokenizer.from_pretrained()
tokenizer = BertTokenizerFast.from_pretrained('./model')
# model = XLNetForSequenceClassification.from_pretrained('./model')
model = BertForSequenceClassification.from_pretrained('./model')
model.to(device)

model.eval()


def predict(content, threshold=.5):
    inputs = tokenizer(content,
                       # return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True, return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    idt = inputs["token_type_ids"].to(device)
    # print(inputs["input_ids"])
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, token_type_ids=idt, attention_mask=mask)
    logits = outputs[0]

    # x = F.sigmoid(logits)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    # print(probs)
    # active_logits = logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)

    # print(active_logits.cpu().detach().numpy()[0])
    # flattened_predictions = active_logits.cpu().detach().numpy()[0]
    flattened_predictions = probs.cpu().detach().numpy()[0]
    # print(flattened_predictions)
    label = {'Claim': [], 'Evidence': [], 'Reasoning': []}
    if flattened_predictions[0] > threshold:
        label['Claim'] = [1, float(flattened_predictions[0])]
    elif flattened_predictions[0] <= threshold:
        label['Claim'] = [0, float(flattened_predictions[0])]
    if flattened_predictions[1] > threshold:
        label['Evidence'] = [1, float(flattened_predictions[1])]
    elif flattened_predictions[1] <= threshold:
        label['Evidence'] = [0, float(flattened_predictions[1])]
    if flattened_predictions[2] > threshold:
        label['Reasoning'] = [1, float(flattened_predictions[2])]
    elif flattened_predictions[2] <= threshold:
        label['Reasoning'] = [0, float(flattened_predictions[2])]
    return label


app = Flask(__name__)  ####
CORS(app)


@app.route("/predict", methods=['GET', 'POST'])
def run():
    if request.method == 'POST':  ####POST
        # data_fz = json.loads(request.get_data().decode('utf-8')) ####get data
        data_fz = request.get_json()
        # print(data_fz)

        if data_fz is not None:
            # data_fz = request.to_dict()
            content = data_fz['content']

        else:
            return jsonify({'Bj': -1, 'Mess': '', 'type': 'Error'})  ####return -1 if no data
    else:
        return jsonify({'Bj': -2, 'Mess': '', 'type': 'Error'})  #### return -2 if not right format

    label = predict(content)

    return jsonify(label)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8008, debug=False)
