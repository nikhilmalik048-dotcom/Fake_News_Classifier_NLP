from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

with open('model_fakenews.pickle', 'rb') as pickle_in:
    pac = pickle.load(pickle_in)

with open('tfid.pickle', 'rb') as tfid:
    tfidf_vectorizer = pickle.load(tfid)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/newscheck')
def newscheck():
    abc = (request.args.get('news') or '').strip()
    if not abc:
        return jsonify(result='FAKE')

    input_data = [abc]
    tfidf_test = tfidf_vectorizer.transform(input_data)
    y_pred = pac.predict(tfidf_test)
    return jsonify(result=str(y_pred[0]).upper())


if __name__ == '__main__':
    app.run(debug=True)
