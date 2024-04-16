import hanlp
from flask import Flask, request, jsonify

app = Flask(__name__)

amr_model_path = "F:/models/amr-eng-zho-xlm-roberta-base_20220412_223756"

amr_parser = hanlp.load(amr_model_path, devices=0)


def amr(toks):
    # 在这里执行根据 toks 生成 amr 字典的逻辑
    amr_result = amr_parser(toks, output_amr=False, language="eng")
    return amr_result


@app.route('/amr_api', methods=['POST'])
def process_amr():
    if request.method == 'POST':
        data = request.get_json()
        toks = data.get('toks', [])
        amr_result = amr(toks)
        return jsonify(amr_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234)
