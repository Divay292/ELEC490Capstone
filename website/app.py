from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

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
        stress_level = request.form['stress_level']
        age = request.form['age']
        occupation = request.form['occupation']
        bmi_category = request.form['bmi_category']
        new_participant = SleepData(stress_level=stress_level, age=age, occupation=occupation, bmi_category=bmi_category)

        try:
            db.session.add(new_participant)
            db.session.commit()
            return redirect('/')
        except:
            return 'errror storing data'
        
    else:
        participant = SleepData.query.order_by(SleepData.id)
        return render_template('index.html', participant=participant)

if __name__ == "__main__":
    app.run(debug=True)