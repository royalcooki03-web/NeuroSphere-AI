from pathlib import Path
import os
import joblib
import pandas as pd

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv


# ============================================================
# NEUROSPHERE AI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / ".env")


# ============================================================
# FLASK CONFIGURATION
# ============================================================

# IMPORTANT:
# Our HTML/CSS/JS files are inside:
#
# app/
# ├── templates/
# │   └── index.html
# │
# └── static/
#     ├── style.css
#     └── app.js
#
# Therefore Flask must be told where these folders are.

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static",
    static_url_path="/static"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = BASE_DIR / "models" / "risk_model.joblib"

DATA_PATH = BASE_DIR / "data" / "student_success_dataset.csv"


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

bundle = None

if MODEL_PATH.exists():

    try:

        bundle = joblib.load(MODEL_PATH)

        print("======================================")
        print("🧠 NeuroSphere AI Model Loaded")
        print("======================================")

    except Exception as error:

        print("⚠️ Model loading error:")
        print(error)

else:

    print("⚠️ ML model not found.")
    print("Run: python train_model.py")


# ============================================================
# ML FEATURES
# ============================================================

FEATURES = [

    "attendance",

    "study_hours",

    "previous_score",

    "assignments",

    "sleep_hours",

    "screen_time",

    "engagement"

]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# HEALTH CHECK API
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "online",

        "model_loaded": MODEL_PATH.exists(),

        "project": "NeuroSphere AI",

        "version": "1.0"

    })


# ============================================================
# DASHBOARD STATISTICS API
# ============================================================

@app.route("/api/stats")
def stats():

    try:

        # Read dataset

        df = pd.read_csv(DATA_PATH)

        # Risk distribution

        risk_distribution = (
            df["risk_level"]
            .value_counts()
            .to_dict()
        )

        return jsonify({

            "students": int(len(df)),

            "average_score": round(
                float(df["previous_score"].mean()),
                1
            ),

            "average_attendance": round(
                float(df["attendance"].mean()),
                1
            ),

            "risk_distribution": risk_distribution

        })

    except Exception as error:

        return jsonify({

            "error": str(error)

        }), 500


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # Check model

        if bundle is None:

            return jsonify({

                "error":
                "Machine Learning model is not loaded. Run python train_model.py"

            }), 500


        # Get JSON request

        payload = request.get_json()

        if not payload:

            return jsonify({

                "error": "No prediction data received."

            }), 400


        # Prepare input data

        row = {

            feature:
            float(payload[feature])

            for feature in FEATURES

        }


        # Create dataframe

        X = pd.DataFrame(

            [row],

            columns=FEATURES

        )


        # Get model

        model = bundle["model"]


        # Prediction

        prediction = model.predict(X)[0]

        risk_level = str(prediction)


        # Prediction probabilities

        probabilities = model.predict_proba(X)[0]

        confidence = float(
            max(probabilities)
        )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        feature_importance = []

        for feature, importance in zip(
            FEATURES,
            model.feature_importances_
        ):

            feature_importance.append({

                "feature": feature,

                "importance":
                round(
                    float(importance),
                    4
                )

            })


        # Sort highest importance first

        feature_importance.sort(

            key=lambda x:
            x["importance"],

            reverse=True

        )


        # ====================================================
        # RESULT MESSAGE
        # ====================================================

        messages = {

            "Low":
            "Strong profile. Keep the current learning routine.",

            "Medium":
            "Some signals need attention. A small improvement plan can help.",

            "High":
            "Multiple risk signals detected. Prioritize attendance, study consistency and assignments."

        }


        message = messages.get(

            risk_level,

            "Review the student's signals."

        )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({

            "risk_level":
            risk_level,

            "confidence":
            round(
                confidence * 100,
                1
            ),

            "message":
            message,

            "signals":
            feature_importance[:5]

        })


    except Exception as error:

        print("Prediction error:")

        print(error)

        return jsonify({

            "error": str(error)

        }), 400


# ============================================================
# DEMO NEUROBOT
# ============================================================

def demo_reply(message):

    message = message.lower().strip()


    # Greeting

    if any(

        word in message

        for word in [
            "hi",
            "hello",
            "hey",
            "hii",
            "helo"
        ]

    ):

        return (
            "Hey! I'm NeuroBot 🤖. "
            "I can help you understand the ML model, "
            "dataset, predictions, dashboard and APIs."
        )


    # Risk

    if "risk" in message:

        return (
            "NeuroSphere AI predicts student risk using "
            "attendance, study hours, previous score, "
            "assignments, sleep, screen time and engagement."
        )


    # Attendance

    if "attendance" in message:

        return (
            "Attendance is an important student-success signal. "
            "Improving attendance can positively affect the "
            "predicted risk level."
        )


    # Machine Learning

    if (

        "machine learning" in message

        or "model" in message

        or "ml" in message

    ):

        return (
            "NeuroSphere AI uses a Random Forest classifier. "
            "The model is trained using the included synthetic "
            "student-success dataset."
        )


    # Dataset

    if (

        "dataset" in message

        or "data" in message

    ):

        return (
            "The project contains 1,200 synthetic student records "
            "with attendance, study hours, previous score, "
            "assignments, sleep, screen time, engagement and risk level."
        )


    # API

    if "api" in message:

        return (
            "NeuroSphere provides REST APIs for health, statistics, "
            "student-risk prediction and AI chat."
        )


    # Internship

    if "internship" in message:

        return (
            "This project is designed as an AIML internship project "
            "combining Machine Learning, Python, Flask, data analytics "
            "and conversational AI."
        )


    # Python

    if "python" in message:

        return (
            "Python powers the backend, Machine Learning model, "
            "data processing and Flask APIs."
        )


    # Random Forest

    if "random forest" in message:

        return (
            "Random Forest is an ensemble Machine Learning algorithm "
            "that combines many decision trees to improve prediction "
            "performance and stability."
        )


    # Default

    return (
        "I can help you with the ML model, dataset, "
        "risk prediction, Random Forest, REST API, "
        "Python or NeuroSphere dashboard. "
        "Ask me anything about the project."
    )


# ============================================================
# AI CHATBOT API
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        payload = request.get_json()

        if not payload:

            return jsonify({

                "reply":
                "Please type a message."

            })


        message = str(

            payload.get(
                "message",
                ""
            )

        ).strip()


        if not message:

            return jsonify({

                "reply":
                "Please type a message."

            })


        # ====================================================
        # OPENAI API
        # ====================================================

        api_key = os.getenv(
            "OPENAI_API_KEY",
            ""
        ).strip()


        # If API key exists → live AI

        if api_key:

            try:

                from openai import OpenAI


                client = OpenAI(

                    api_key=api_key

                )


                model_name = os.getenv(

                    "OPENAI_MODEL",

                    "gpt-4o-mini"

                )


                response = (

                    client
                    .chat
                    .completions
                    .create(

                        model=model_name,

                        messages=[

                            {

                                "role":
                                "system",

                                "content":

                                (
                                    "You are NeuroBot, "
                                    "the professional AI assistant "
                                    "inside NeuroSphere AI. "
                                    "NeuroSphere is an AIML student "
                                    "success prediction project. "
                                    "Explain Machine Learning concepts "
                                    "clearly and professionally."
                                )

                            },

                            {

                                "role":
                                "user",

                                "content":
                                message

                            }

                        ],

                        temperature=0.5

                    )

                )


                reply = (

                    response
                    .choices[0]
                    .message
                    .content

                )


                return jsonify({

                    "reply":
                    reply,

                    "mode":
                    "live"

                })


            except Exception as api_error:

                print("OpenAI API error:")

                print(api_error)

                # If live API fails,
                # automatically use demo chatbot

                return jsonify({

                    "reply":
                    demo_reply(message),

                    "mode":
                    "demo"

                })


        # ====================================================
        # DEMO MODE
        # ====================================================

        return jsonify({

            "reply":
            demo_reply(message),

            "mode":
            "demo"

        })


    except Exception as error:

        return jsonify({

            "reply":
            "NeuroBot encountered an error.",

            "error":
            str(error)

        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "error":
        "Page not found",

        "status":
        404

    }), 404


@app.errorhandler(500)
def server_error(error):

    return jsonify({

        "error":
        "Internal server error",

        "status":
        500

    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")
    print("==============================================")
    print("        🚀 NEUROSPHERE AI")
    print("==============================================")
    print("🧠 Machine Learning: READY")
    print("🤖 NeuroBot: READY")
    print("📊 Dataset: READY")
    print("🌐 Flask Server: STARTING")
    print("==============================================")
    print("")

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )