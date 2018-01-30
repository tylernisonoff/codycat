import os
import time
import json
from flask import Flask, render_template, request
from fastai.conv_learner import *

app = Flask(__name__, static_url_path='/static')
print("Starting flask app...")

PATH = 'data/shopstyle/'
f_model = resnet34
label_csv = f'{PATH}prod_train.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)
sz = 128

def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train', label_csv, tfms=tfms, suffix='.jpg', val_idxs=val_idxs, test_name='test')

data = get_data(sz)
learn = ConvLearner.pretrained(f_model, data)

def load_model():
    print("Loading model...")
    learn.load(f'{sz}')
    learn.precompute=False

load_model()

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
    tuples = list(zip(data.classes, test_preds[0]))
    #convert to a float
    tuples = list(map(lambda x: (x[0], float(x[1])), tuples))
    #throw away anything less than 0.1
    result = [item for item in tuples if item[1] > 0.1]
    #return up to 5 categories
    return json.dumps(dict(result[:5])
