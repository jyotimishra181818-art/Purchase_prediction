# Customer Purchase Prediction

Flask + SQLite + scikit-learn (logistic regression) + Claude AI assistant.

## Run
    python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your-key                    # optional; Windows: set ANTHROPIC_API_KEY=...
    python app.py                                        # http://localhost:5000

Register an account, open Customers -> "Load 80 demo customers", then explore Dashboard / Predict / AI Assistant.
Without an API key the assistant uses built-in rule-based answers.

## Structure
    app.py            routes, auth, SQLite, model training, AI chat
    templates/        Jinja pages      static/style.css
    data.db           created on first run

## Production notes
Set SECRET_KEY, run behind gunicorn + HTTPS (`gunicorn app:app`), and add login rate limiting.

## PostgreSQL (optional)
    docker compose up -d db
    pip install psycopg2-binary
    export DATABASE_URL=postgresql://predictor:predictor@localhost:5432/predictor
    python app.py
Without `DATABASE_URL` the app uses SQLite (`data.db`).

## CSV
Customers page -> **Import CSV** (columns: name, age, income, visits, minutes, cart, past, email_opened, purchased)
and **Export CSV** (same columns plus `purchase_probability`; exports re-import cleanly).
