# NeuroSphere AI
Professional AIML internship project: ML student-risk prediction + animated dashboard + NeuroBot + dataset + REST API.

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python train_model.py
python app.py
```
Open http://127.0.0.1:5000

### Live AI
Put your OpenAI API key in `.env`:
`OPENAI_API_KEY=your_key_here`
Without a key, NeuroBot runs in demo mode.

### Included
- 1,200-record synthetic dataset
- trained Random Forest model
- animated glassmorphism dashboard
- Canvas particle/3D-style background
- AI chatbot
- prediction REST API
- health + stats APIs
- responsive UI
