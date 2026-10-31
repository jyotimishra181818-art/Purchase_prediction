"""Customer Purchase Prediction - Flask + SQLite + scikit-learn + Claude assistant."""
import csv, io, os, random, secrets, sqlite3
from functools import wraps
from flask import (Flask, Response, g, jsonify, redirect, render_template, request,
                   session, url_for, flash, abort)
from werkzeug.security import check_password_hash, generate_password_hash
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE, "data.db"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")  # set to use PostgreSQL, else SQLite
PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))
FEATURES = [("age", "Age"), ("income", "Income (k$)"), ("visits", "Visits / month"),
            ("minutes", "Avg minutes on site"), ("cart", "Items in cart"),
            ("past", "Past orders"), ("email_opened", "Opened email (0/1)")]
KEYS = [k for k, _ in FEATURES]
MIN_LABELED = 12

app = Flask(__name__)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  MAX_CONTENT_LENGTH=2 * 1024 * 1024)

def _secret():
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = os.path.join(BASE, ".secret")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    return open(path).read().strip()
app.secret_key = _secret()

# ---------------------------------------------------------------- database
class _PG:
    """Thin wrapper so the same '?'-style SQL runs on PostgreSQL."""
    def __init__(self, con): self.con = con
    def execute(self, sql, params=()):
        import psycopg2.extras
        cur = self.con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql.replace("?", "%s"), params)
        return cur
    def commit(self): self.con.commit()
    def close(self): self.con.close()

def db():
    if "db" not in g:
        if PG:
            import psycopg2
            g.db = _PG(psycopg2.connect(DATABASE_URL))
        else:
            g.db = sqlite3.connect(DB_PATH)
            g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()

def init_db():
    pk, real, idt = ("BIGSERIAL PRIMARY KEY", "DOUBLE PRECISION", "BIGINT") if PG else ("INTEGER PRIMARY KEY", "REAL", "INTEGER")
    script = f"""
    CREATE TABLE IF NOT EXISTS users(
        id {pk}, name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS customers(
        id {pk}, user_id {idt} NOT NULL REFERENCES users(id),
        name TEXT NOT NULL, age {real}, income {real}, visits {real}, minutes {real},
        cart {real}, past {real}, email_opened {real}, purchased INTEGER);
    CREATE INDEX IF NOT EXISTS idx_cust_user ON customers(user_id);"""
    if PG:
        import psycopg2
        con = psycopg2.connect(DATABASE_URL); cur = con.cursor(); cur.execute(script)
    else:
        con = sqlite3.connect(DB_PATH); con.executescript(script)
    con.commit(); con.close()

# -------------------------------------------------------------------- auth
def login_required(f):
    @wraps(f)
    def wrap(*a, **kw):
        if "uid" not in session:
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrap

def csrf():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]
app.jinja_env.globals["csrf"] = csrf

@app.before_request
def check_csrf():
    if request.method == "POST":
        tok = request.form.get("csrf") or request.headers.get("X-CSRF")
        if not tok or tok != session.get("csrf"):
            abort(400, "Bad CSRF token")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        pw = request.form["password"]
        if not name or "@" not in email or len(pw) < 6:
            flash("Enter a name, valid email and a password of 6+ characters.")
        elif db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            flash("An account with that email already exists.")
        else:
            cur = db().execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?) RETURNING id",
                               (name, email, generate_password_hash(pw)))
            new_id = cur.fetchone()["id"]
            db().commit()
            session.clear(); session["uid"] = new_id; session["name"] = name
            return redirect(url_for("dashboard"))
    return render_template("auth.html", mode="register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db().execute("SELECT * FROM users WHERE email=?",
                         (request.form["email"].strip().lower(),)).fetchone()
        if u and check_password_hash(u["password_hash"], request.form["password"]):
            session.clear(); session["uid"] = u["id"]; session["name"] = u["name"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.")
    return render_template("auth.html", mode="login")

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))

# ------------------------------------------------------------------- model
MODELS = {}  # user_id -> trained bundle (cache, invalidated on data change)

def customers(uid):
    return db().execute("SELECT * FROM customers WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()

def vec(r):
    return [float(r[k] or 0) for k in KEYS]

def get_model(uid):
    if uid in MODELS:
        return MODELS[uid]
    labeled = [r for r in customers(uid) if r["purchased"] in (0, 1)]
    y = [r["purchased"] for r in labeled]
    bundle = None
    if len(labeled) >= MIN_LABELED and len(set(y)) == 2:
        X = [vec(r) for r in labeled]
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        except ValueError:
            Xtr, Xte, ytr, yte = X, X, y, y
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), ytr)
        acc = float(clf.score(sc.transform(Xte), yte))
        sc_all = StandardScaler().fit(X)
        clf_all = LogisticRegression(max_iter=1000).fit(sc_all.transform(X), y)  # final model on all data
        bundle = {"scaler": sc_all, "clf": clf_all, "acc": acc, "n": len(labeled),
                  "weights": sorted(zip([l for _, l in FEATURES], map(float, clf_all.coef_[0])),
                                    key=lambda t: -abs(t[1]))}
    MODELS[uid] = bundle
    return bundle

def score_rows(uid, rows):
    m = get_model(uid)
    if not m or not rows:
        return {}
    p = m["clf"].predict_proba(m["scaler"].transform([vec(r) for r in rows]))[:, 1]
    return {r["id"]: float(x) for r, x in zip(rows, p)}

def segment(p):
    if p is None: return ("-", "")
    return ("High", "hi") if p >= .7 else ("Medium", "md") if p >= .4 else ("Low", "lo")
app.jinja_env.globals["segment"] = segment

# ------------------------------------------------------------------- pages
@app.route("/")
@login_required
def dashboard():
    uid = session["uid"]; rows = customers(uid); m = get_model(uid); sc = score_rows(uid, rows)
    labeled = [r for r in rows if r["purchased"] in (0, 1)]
    conv = round(100 * sum(r["purchased"] for r in labeled) / len(labeled)) if labeled else None
    prospects = sorted([(r, sc[r["id"]]) for r in rows if r["purchased"] is None and r["id"] in sc],
                       key=lambda t: -t[1])
    return render_template("dashboard.html", total=len(rows), conv=conv, model=m,
                           prospects=prospects[:10], hot=sum(1 for _, p in prospects if p >= .7),
                           need=MIN_LABELED)

@app.route("/customers", methods=["GET", "POST"])
@login_required
def customers_page():
    uid = session["uid"]
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Name is required.")
        else:
            buy = request.form.get("purchased", "")
            vals = []
            for k in KEYS:
                try: vals.append(float(request.form.get(k) or 0))
                except ValueError: vals.append(0.0)
            db().execute(f"INSERT INTO customers(user_id,name,{','.join(KEYS)},purchased) VALUES(?,?,{','.join('?'*len(KEYS))},?)",
                         (uid, name, *vals, int(buy) if buy in ("0", "1") else None))
            db().commit(); MODELS.pop(uid, None)
        return redirect(url_for("customers_page"))
    rows = customers(uid)
    return render_template("customers.html", rows=rows, scores=score_rows(uid, rows), features=FEATURES)

@app.post("/customers/<int:cid>/delete")
@login_required
def delete_customer(cid):
    db().execute("DELETE FROM customers WHERE id=? AND user_id=?", (cid, session["uid"]))
    db().commit(); MODELS.pop(session["uid"], None)
    return redirect(url_for("customers_page"))

@app.post("/customers/clear")
@login_required
def clear_customers():
    db().execute("DELETE FROM customers WHERE user_id=?", (session["uid"],))
    db().commit(); MODELS.pop(session["uid"], None)
    return redirect(url_for("customers_page"))

@app.post("/customers/demo")
@login_required
def demo():
    import math
    names = "Ava Liam Noah Mia Eli Zoe Ravi Priya Arjun Sara Omar Lena Kai Nora Diya Ben".split()
    for i in range(80):
        r = dict(age=random.randint(18, 67), income=random.randint(25, 125), visits=random.randint(0, 14),
                 minutes=round(random.uniform(0, 20), 1), cart=random.randint(0, 5),
                 past=random.randint(0, 9), email_opened=random.randint(0, 1))
        z = (-3.2 + .25*r["visits"] + .12*r["minutes"] + .6*r["cart"] + .2*r["past"]
             + .9*r["email_opened"] + .01*r["income"] + random.uniform(-1, 1))
        buy = (1 if random.random() < 1/(1+math.exp(-z)) else 0) if i < 65 else None
        db().execute(f"INSERT INTO customers(user_id,name,{','.join(KEYS)},purchased) VALUES(?,?,{','.join('?'*len(KEYS))},?)",
                     (session["uid"], f"{names[i % 16]} {chr(65 + i % 26)}.", *[r[k] for k in KEYS], buy))
    db().commit(); MODELS.pop(session["uid"], None)
    return redirect(url_for("customers_page"))

@app.route("/customers/export")
@login_required
def export_csv():
    rows = customers(session["uid"]); sc = score_rows(session["uid"], rows)
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["name", *KEYS, "purchased", "purchase_probability"])
    for r in rows:
        name = r["name"]
        if name.startswith(("=", "+", "-", "@", "\t", "\r")):  # block spreadsheet formula injection
            name = "'" + name
        p = sc.get(r["id"])
        w.writerow([name, *[r[k] for k in KEYS], "" if r["purchased"] is None else r["purchased"],
                    "" if p is None else round(p, 4)])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=customers.csv"})

@app.post("/customers/import")
@login_required
def import_csv():
    f = request.files.get("file")
    if not f:
        flash("Choose a CSV file."); return redirect(url_for("customers_page"))
    n = 0
    for row in csv.DictReader(io.StringIO(f.read().decode("utf-8-sig", "ignore"))):
        if n >= 5000 or not (row.get("name") or "").strip(): continue
        vals = []
        for k in KEYS:
            try: vals.append(float(row.get(k) or 0))
            except ValueError: vals.append(0.0)
        p = (row.get("purchased") or "").strip()
        db().execute(f"INSERT INTO customers(user_id,name,{','.join(KEYS)},purchased) VALUES(?,?,{','.join('?'*len(KEYS))},?)",
                     (session["uid"], row["name"].strip(), *vals, int(p) if p in ("0", "1") else None))
        n += 1
    db().commit(); MODELS.pop(session["uid"], None)
    flash(f"Imported {n} customers.")
    return redirect(url_for("customers_page"))

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    m = get_model(session["uid"]); result = None; vals = {k: 0 for k in KEYS}
    if request.method == "POST" and m:
        for k in KEYS:
            try: vals[k] = float(request.form.get(k) or 0)
            except ValueError: vals[k] = 0
        result = float(m["clf"].predict_proba(m["scaler"].transform([[vals[k] for k in KEYS]]))[0, 1])
    return render_template("predict.html", model=m, result=result, vals=vals, features=FEATURES, need=MIN_LABELED)

# ------------------------------------------------------------ AI assistant
@app.route("/assistant")
@login_required
def assistant():
    return render_template("assistant.html", live=bool(os.environ.get("ANTHROPIC_API_KEY")))

def context_summary(uid):
    rows = customers(uid); m = get_model(uid); sc = score_rows(uid, rows)
    top = sorted([(r["name"], sc[r["id"]]) for r in rows if r["purchased"] is None and r["id"] in sc], key=lambda t: -t[1])[:10]
    s = f"Customers: {len(rows)}. "
    s += (f"Model: logistic regression, holdout accuracy {m['acc']:.0%}, trained on {m['n']} labeled customers. "
          f"Standardized weights: " + "; ".join(f"{l} {w:+.2f}" for l, w in m["weights"]) + ". ") if m else "Model not trained yet. "
    s += "Top unpurchased prospects: " + (", ".join(f"{n} {p:.0%}" for n, p in top) or "none") + "."
    return s, top, m

def fallback_answer(question, top, m):
    q = question.lower()
    if not m:
        return f"I need at least {MIN_LABELED} customers with known outcomes before I can predict. Add customers or load demo data."
    if any(w in q for w in ("driver", "why", "factor", "important", "drive")):
        return "Biggest drivers (standardized weights):\n" + "\n".join(f"- {l}: {w:+.2f}" for l, w in m["weights"][:5])
    if any(w in q for w in ("campaign", "market", "offer", "email")):
        return ("Suggested plan: High (70%+) -> personal outreach or time-limited offer; Medium (40-70%) -> reminder email "
                "or small incentive; Low (<40%) -> low-cost automated nurture only.")
    return "Contact these first:\n" + "\n".join(f"- {n}: {p:.0%}" for n, p in top[:5]) if top else "No unscored prospects. Add customers with unknown outcome."

@app.post("/api/chat")
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    history = [{"role": h["role"], "content": str(h["content"])[:2000]}
               for h in data.get("history", [])[-10:] if h.get("role") in ("user", "assistant")]
    question = str(data.get("message", ""))[:2000].strip()
    if not question:
        return jsonify(error="Empty message"), 400
    ctx, top, m = context_summary(session["uid"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            resp = anthropic.Anthropic().messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"), max_tokens=700,
                system=("You are the assistant inside a Customer Purchase Prediction tool. Be concise and practical. "
                        "Use only this data summary; never invent customers.\n" + ctx),
                messages=history + [{"role": "user", "content": question}])
            return jsonify(reply="".join(b.text for b in resp.content if b.type == "text"))
        except Exception as e:
            app.logger.exception(e)
            return jsonify(reply="The AI service failed, using built-in answer:\n" + fallback_answer(question, top, m))
    return jsonify(reply=fallback_answer(question, top, m))

init_db()
if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
