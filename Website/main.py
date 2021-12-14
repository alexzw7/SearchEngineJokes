
from flask import Flask, render_template, redirect, request
from pandas.core import base
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os
import search

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
index = search.loadIndex()
docs = search.readDocs()

baseline_pipe = search.bm_pipe(index)
rm_pipe = search.rm_pipe(index)
l2r_pipe = search.fit_l2r(index)

class BasicForm(FlaskForm):
    ids = StringField("ID",validators=[DataRequired()])
    submit = SubmitField("Submit")

@app.route("/",methods =['POST','GET'])
def main():
    form = BasicForm()
    return render_template("index.html",form=form)

@app.route("/search",methods =['POST'])
def search_results():
    query = request.form['query']
    if len(query) == 0:
        return redirect('/')
    results = search.search(query, l2r_pipe)
    top = []
    for result in results:
        top.append(docs[result])
    return render_template("search.html", query=query, results=top)

if __name__ == "__main__":
    app.run(debug=False)