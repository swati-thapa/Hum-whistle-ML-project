
import numpy as np
from flask import Flask, request, render_template
import pickle
import librosa
import pandas as pd




model_rms = pickle.load(open('Pickle_rms.pkl', 'rb'))
model_zcr = pickle.load(open('Pickle_zcr.pkl', 'rb'))
model_spec_cent = pickle.load(open('Pickle_spec_cent.pkl', 'rb'))
model_all = pickle.load(open('Pickle_all_three.pkl', 'rb'))

ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
#@app.route('/find' , methods = ['GET' , 'POST'])
def find():
    if request.method == 'GET':
        return "HELLO"

@app.route('/' , methods = ['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):

            audio_data, sr = librosa.load(file)

            rms = librosa.feature.rms(audio_data)[0]
            rms_test_df = pd.DataFrame(rms)
            rms_input = rms_test_df.T
            df= pd.DataFrame(0, index=np.arange(1), columns=np.arange(2651))
            final_df = rms_input + df
            final_df.fillna(0, inplace=True)


            prediction_rms= model_rms.predict(final_df)
            rms_value=prediction_rms[0]

            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            zcr_test_df = pd.DataFrame(zcr)
            zcr_input = zcr_test_df.T
            d = pd.DataFrame(0, index=np.arange(1), columns=np.arange(2651))
            final_df = zcr_input + d
            final_df.fillna(0, inplace=True)


            prediction_zcr = model_zcr.predict(final_df)
            zcr_value = prediction_zcr[0]

            spec_cent = librosa.feature.spectral_centroid(audio_data)[0]
            spec_cent_test_df = pd.DataFrame(spec_cent)
            spec_cent_test_df_input = spec_cent_test_df.T
            df = pd.DataFrame(0, index=np.arange(1), columns=np.arange(2651))
            final_df = spec_cent_test_df_input + df
            final_df.fillna(0, inplace=True)

            prediction_spec_cent = model_spec_cent.predict(final_df)
            spec_cent_value = prediction_spec_cent[0]

            prediction= model_all.predict([[rms_value,zcr_value,spec_cent_value]])


            if prediction[0]== True:
                output = "Harry Potter"
            else:
                output="StarWars"

    return render_template('index.html', prediction_text=output)


if __name__ == "__main__":
  #app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)
  app.run(port=5001, debug=True, use_reloader=True)