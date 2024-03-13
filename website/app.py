from flask import Flask, render_template, url_for, request, redirect, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['UPLOAD_FOLDER'] = '.'  # Current directory

class SleepData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stress_level = db.Column(db.String(10), nullable=False) 
    age = db.Column(db.Integer, nullable=False)
    occupation = db.Column(db.String(100), nullable=False)
    bmi_category = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return'<Participant %r>' % self.id


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        file = request.files['Data']
        stress_level = request.form['stress_level']
        age = request.form['age']
        occupation = request.form['occupation']
        bmi_category = request.form['bmi_category']

        # Save file
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Create a new participant instance
        new_participant = SleepData(
            stress_level=stress_level, 
            age=age, 
            occupation=occupation, 
            bmi_category=bmi_category
        )

        try:
            # Add the new participant to the database
            db.session.add(new_participant)
            db.session.commit()
            return redirect(url_for('result'))
        except:
            return 'Error storing data'

    else:
        participants = SleepData.query.order_by(SleepData.id).all()
        return render_template('index.html', participants=participants)

@app.route('/result', methods=['GET'])
def result():
    # file_path = os.path.join(app.config['uploads'], 'sleepScoreOutputs.txt')
    with open("sleepScoreOutputs.txt", 'r') as file:
        numbers = [float(line.strip()) for line in file.readlines()]
        average = np.mean(numbers)
    average = "{:.2f}".format(float(average))

    return render_template('result.html', average=average)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(port=8000, debug=True)