from flask import Flask, render_template, request
import pickle

# Load the tokenizer and model
tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

# Initialize the Flask application
app = Flask(__name__)

# Define the homepage route to render HTML
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        email_text = request.form.get('email-content')
        tokenize_email = tokenizer.transform([email_text])  # Ensure the input is in the correct format
        predictions = model.predict(tokenize_email)
        predictions = 1 if predictions[0] == 1 else -1  # Access the first element
        return render_template("index.html", predictions=predictions, email_text=email_text)

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
