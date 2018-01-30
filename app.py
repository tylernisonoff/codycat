import os
import time
import json
from flask import Flask, render_template, request, url_for
from fastai.conv_learner import *

app = Flask(__name__, static_url_path='/data')
app._static_folder = 'data'
print("Starting flask app...")

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PATH = os.path.join(CURRENT_DIR, 'data')
f_model = resnet34
sz = 224

def get_data(sz):
    tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, test_name='test')
    return data

data = get_data(sz)
learn = ConvLearner.pretrained(f_model, data)

def load_model():
    print("Loading model...")
    learn.load('codycat_2')
    learn.precompute=False

load_model()

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     app._static_folder, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route('/', methods=['GET'])
def home():
    return '''
<h1>Upload a jpg or png file of Gal or Ibu.  I'll tell you who I think it is!<h1>
<form action='/upload' method=post enctype=multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
</form>
'''
@app.route('/upload', methods=['POST'])
def upload_file():
    print("cleaning test dir")
    for root, dirs, files in os.walk(PATH+'/test'):
        for f in files:
            os.unlink(os.path.join(root, f))
    file = request.files['file']
    f = os.path.join(PATH+"/test", 'placeholder.jpg')
    file.save(f)


    start_time = time.process_time()
    load_model();
    test_preds = learn.predict(is_test=True)
    end_time = time.process_time()
    print("Elapsed time: %.9f" % (end_time-start_time))
    print(test_preds)
    probs = np.exp(test_preds)[0]
    print(probs)

    guess = "Gal"
    prob_gal = probs[0]
    display_prob = prob_gal
    if prob_gal < 0.5:
        guess = "Ibu"
        display_prob = 1 - display_prob
    #return up to 5 categories

    display_percent_str = "{0:.3f}".format(display_prob * 100)
    return '''
<div>
    <h1>{guess} : {percent}%</h1>
    <img src="{src}" height="500" >
</div>
'''.format(guess=guess, src=dated_url_for('static', filename='test/placeholder.jpg'), percent= display_percent_str)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=True)
