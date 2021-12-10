def run_server():
    import io
    from flask import Flask, render_template, request, send_file
    # from flask_uploads import UploadSet, configure_uploads, IMAGES
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import transformers
    from model import handle
    import os
    import random
    import string
    import numpy as np
    from model import handle
    import time
    print('main')
    app = Flask(__name__)
    @app.route('/')
    def my_form():
        print('post')
        return render_template('form.html')
    @app.route('/', methods=['POST'])
    def my_form_post():
        print('ok')
        text = request.form['text']
        texts=handle(text)
        print(texts)
        texts=((texts,))
        print(texts)
        return render_template('form1.html',texts=texts)

    app.run(host = '0.0.0.0',debug=True)
if __name__ == "__main__":
    run_server()