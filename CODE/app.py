import os, pickle, sqlite3, json, csv, io
from datetime import datetime
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify, Response)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH   = os.path.join(BASE_DIR, "agri.db")

app = Flask(__name__)


# ── Core utility routes ───────────────────────────────────────────────────────
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/api/season_from_month/<int:month>")
def season_from_month(month):
    from flask import jsonify as _j
    MONTH_TO_SEASON = {1:"Rabi",2:"Rabi",3:"Rabi",4:"Summer",5:"Summer",
        6:"Kharif",7:"Kharif",8:"Kharif",9:"Kharif",10:"Kharif",11:"Rabi",12:"Rabi"}
    return _j({"season": MONTH_TO_SEASON.get(month, "Kharif"), "month": month})
# ─────────────────────────────────────────────────────────────────────────────

app.secret_key = "agri_secret_2024_v2"

def load_models():
    clf       = pickle.load(open(os.path.join(MODEL_DIR,"crop_recommender.pkl"),"rb"))
    reg       = pickle.load(open(os.path.join(MODEL_DIR,"yield_predictor.pkl"),"rb"))
    le_crop   = pickle.load(open(os.path.join(MODEL_DIR,"le_crop.pkl"),"rb"))
    le_season = pickle.load(open(os.path.join(MODEL_DIR,"le_season.pkl"),"rb"))
    le_state  = pickle.load(open(os.path.join(MODEL_DIR,"le_state.pkl"),"rb"))
    scaler    = pickle.load(open(os.path.join(MODEL_DIR,"scaler.pkl"),"rb"))
    scaler_y  = pickle.load(open(os.path.join(MODEL_DIR,"scaler_yield.pkl"),"rb"))
    meta      = pickle.load(open(os.path.join(MODEL_DIR,"meta.pkl"),"rb"))
    return clf, reg, le_crop, le_season, le_state, scaler, scaler_y, meta

clf, reg, le_crop, le_season, le_state, scaler, scaler_y, META = load_models()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db(); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL, role TEXT DEFAULT 'farmer',
        state TEXT, district TEXT, phone TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS equipment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        owner_id INTEGER, name TEXT NOT NULL, category TEXT,
        description TEXT, daily_rate REAL, state TEXT, district TEXT,
        is_available INTEGER DEFAULT 1, photo_url TEXT,
        FOREIGN KEY (owner_id) REFERENCES users(id))""")
    # Migrate: add photo_url if upgrading from older DB
    try: c.execute("ALTER TABLE equipment ADD COLUMN photo_url TEXT")
    except: pass
    c.execute("""CREATE TABLE IF NOT EXISTS bookings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_id INTEGER, equipment_id INTEGER,
        start_date TEXT, end_date TEXT, total_cost REAL,
        status TEXT DEFAULT 'pending', notes TEXT,
        pickup_address TEXT, pickup_district TEXT, pickup_state TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (farmer_id) REFERENCES users(id),
        FOREIGN KEY (equipment_id) REFERENCES equipment(id))""")
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, crop TEXT, season TEXT, state TEXT,
        area REAL, rainfall REAL, fertilizer REAL, pesticide REAL,
        predicted_yield REAL, predicted_low REAL, predicted_high REAL,
        risk_level TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        booking_id INTEGER UNIQUE, farmer_id INTEGER,
        equipment_id INTEGER, provider_id INTEGER,
        rating INTEGER, review TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (booking_id) REFERENCES bookings(id))""")

    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO users (name,email,password,role,state,district,phone) VALUES (?,?,?,?,?,?,?)", [
            ("Ravi Kumar","farmer@demo.com","demo123","farmer","Karnataka","Bangalore Rural","9876543210"),
            ("AgriRent Co.","provider@demo.com","demo123","provider","Karnataka","Tumkur","9876500001"),
            ("Tamil Farmer","tamil@demo.com","demo123","farmer","Tamil Nadu","Coimbatore","9876511111"),
            ("Admin User","admin@demo.com","admin123","admin","Karnataka","Bangalore","9876522222"),
        ])
    c.execute("SELECT COUNT(*) FROM equipment")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO equipment (owner_id,name,category,description,daily_rate,state,district,is_available) VALUES (?,?,?,?,?,?,?,?)",[
            (2,"Tractor (55HP)","Tractor","Mahindra 575 DI, suitable for all soil types",1500,"Karnataka","Tumkur",1),
            (2,"Rotavator","Tillage","Tractor-mounted rotary tiller, 7-ft width",700,"Karnataka","Tumkur",1),
            (2,"Power Sprayer","Spraying","15L battery-powered backpack sprayer",200,"Karnataka","Tumkur",1),
            (2,"Seed Drill","Sowing","Multi-crop seed drill, 9-row",800,"Karnataka","Bangalore Rural",1),
            (2,"Combine Harvester","Harvesting","Self-propelled combine harvester",4500,"Karnataka","Tumkur",1),
            (2,"Drip Irrigation Kit","Irrigation","Complete drip kit for 1 acre",350,"Tamil Nadu","Coimbatore",1),
            (2,"Mini Tractor","Tractor","21HP compact tractor for small farms",900,"Tamil Nadu","Coimbatore",1),
        ])
    conn.commit(); conn.close()

init_db()

# ── Helpers ───────────────────────────────────────────────────────────────────
def current_user():
    uid = session.get("user_id")
    if not uid: return None
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return user

def smart_alerts(season, state, rainfall, area, fertilizer, crop=None, pesticide=0):
    alerts = []
    if rainfall < 500:
        alerts.append({"type":"warning","icon":"💧","title":"Low Rainfall","msg":f"{rainfall}mm below optimal. Consider drip irrigation or drought-resistant varieties."})
    elif rainfall > 3500:
        alerts.append({"type":"danger","icon":"🌧️","title":"Flood Risk","msg":f"{rainfall}mm is very high. Risk of waterlogging — ensure proper drainage."})
    if area > 0:
        fph = fertilizer / area
        if fph > 500:
            alerts.append({"type":"warning","icon":"⚗️","title":"Excess Fertilizer","msg":f"{fph:.0f} kg/ha is too high. May cause soil acidification and runoff pollution."})
        elif fph < 50:
            alerts.append({"type":"info","icon":"🌿","title":"Low Fertilizer","msg":f"{fph:.0f} kg/ha is low. Get a soil test to optimise inputs."})
        pph = pesticide / area if pesticide > 0 else 0
        if pph > 5:
            alerts.append({"type":"danger","icon":"☠️","title":"High Pesticide","msg":f"{pph:.1f} kg/ha is dangerously high. Risk to soil microbiome and human health."})
        elif pph > 2:
            alerts.append({"type":"warning","icon":"🧪","title":"Elevated Pesticide","msg":f"{pph:.1f} kg/ha is above recommended levels. Consider integrated pest management."})
    
    if not alerts:
        alerts.append({"type":"success","icon":"✅","title":"All Clear","msg":"Parameters look optimal for the selected season and region."})
    return alerts

def yield_confidence_interval(X_input, estimators, rainfall=None, fertilizer=None, area=None):
    preds = np.array([tree.predict(X_input)[0] for tree in estimators])
    mean = np.mean(preds); std = np.std(preds)
    low = max(0, mean - 1.96*std); high = mean + 1.96*std
    cv = (std/mean*100) if mean > 0 else 0

    # ── Risk based purely on farming conditions ───────────────────────────────
    # RF tree-level CV is always high (60-100%+) so is NOT used for risk rating.
    # Risk is determined by how far inputs deviate from optimal agri ranges.
    risk_score = 0

    if rainfall is not None:
        if rainfall < 200 or rainfall > 3500:    risk_score += 3  # very severe
        elif rainfall < 400 or rainfall > 2800:  risk_score += 2  # severe
        elif rainfall < 600 or rainfall > 2200:  risk_score += 1  # moderate

    if fertilizer is not None and area is not None and area > 0:
        fph = fertilizer / area
        if fph < 20 or fph > 700:   risk_score += 3
        elif fph < 50 or fph > 500: risk_score += 2
        elif fph < 80 or fph > 400: risk_score += 1

    risk = "Low" if risk_score <= 1 else "Medium" if risk_score <= 3 else "High"
    risk_color = "#27ae60" if risk=="Low" else "#f39c12" if risk=="Medium" else "#e74c3c"
    return {"mean":round(mean,3),"low":round(low,3),"high":round(high,3),"cv":round(cv,1),"risk":risk,"risk_color":risk_color}

@app.context_processor
def inject_user():
    return dict(current_user=current_user(), crops=META["crops"],
                seasons=META["seasons"], states=META["states"])

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    conn = get_db()
    stats = {
        "crops": len(META["crops"]), "states": len(META["states"]),
        "r2": META["regressor_r2"],
        "bookings": conn.execute("SELECT COUNT(*) FROM bookings WHERE status='approved'").fetchone()[0],
        "equipment": conn.execute("SELECT COUNT(*) FROM equipment WHERE is_available=1").fetchone()[0],
        "farmers": conn.execute("SELECT COUNT(*) FROM users WHERE role='farmer'").fetchone()[0],
    }
    conn.close()
    return render_template("index.html", stats=stats)

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email=request.form["email"]; pwd=request.form["password"]
        conn=get_db()
        user=conn.execute("SELECT * FROM users WHERE email=? AND password=?",(email,pwd)).fetchone()
        conn.close()
        if user:
            session["user_id"]=user["id"]
            flash(f"Welcome back, {user['name']}! 🌾","success")
            if user["role"]=="admin": return redirect(url_for("admin"))
            if user["role"]=="provider": return redirect(url_for("dashboard"))
            return redirect(url_for("index"))
        flash("Invalid credentials.","danger")
    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        conn=get_db()
        try:
            conn.execute("INSERT INTO users (name,email,password,role,state,district,phone) VALUES (?,?,?,?,?,?,?)",
                (request.form["name"],request.form["email"],request.form["password"],
                 request.form.get("role","farmer"),request.form.get("state",""),
                 request.form.get("district",""),request.form.get("phone","")))
            conn.commit(); flash("Account created! Please login.","success")
            return redirect(url_for("login"))
        except: flash("Email already registered.","danger")
        finally: conn.close()
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear(); flash("Logged out.","info"); return redirect(url_for("index"))

@app.route("/recommend", methods=["GET","POST"])
def recommend():
    result=None; form_data={}
    if request.method=="POST":
        try:
            season=request.form["season"]; state=request.form["state"]
            area=float(request.form["area"]); rainfall=min(max(float(request.form["rainfall"]),0),5000)
            fertilizer=float(request.form["fertilizer"]); pesticide=float(request.form.get("pesticide",0))
            form_data=dict(season=season,state=state,area=area,rainfall=rainfall,fertilizer=fertilizer,pesticide=pesticide)
            se=le_season.transform([season])[0]; ste=le_state.transform([state])[0]
            Xi=scaler.transform([[se,ste,area,rainfall,fertilizer,pesticide]])
            proba=clf.predict_proba(Xi)[0]
            top5=[(le_crop.classes_[i],round(proba[i]*100,1)) for i in np.argsort(proba)[::-1][:5]]
            result={"top5":top5,"alerts":smart_alerts(season,state,rainfall,area,fertilizer,pesticide=pesticide)}
        except Exception as e: flash(f"Error: {e}","danger")
    return render_template("recommend.html", result=result, form_data=form_data)

@app.route("/predict", methods=["GET","POST"])
def predict():
    result=None; form_data={}
    if request.method=="POST":
        try:
            crop=request.form["crop"]; season=request.form["season"]; state=request.form["state"]
            area=float(request.form["area"]); rainfall=min(5000.0, max(0.0, float(request.form["rainfall"])))
            fertilizer=float(request.form["fertilizer"]); pesticide=float(request.form.get("pesticide",0))
            form_data=dict(crop=crop,season=season,state=state,area=area,rainfall=rainfall,fertilizer=fertilizer,pesticide=pesticide)
            ce=le_crop.transform([crop])[0]; se=le_season.transform([season])[0]; ste=le_state.transform([state])[0]
            Xi=scaler_y.transform([[se,ste,ce,area,rainfall,fertilizer,pesticide]])
            ci=yield_confidence_interval(Xi, reg.estimators_, rainfall=rainfall, fertilizer=fertilizer, area=area)
            result={"crop":crop,"area":area,
                "yield_per_ha":ci["mean"],"total_yield":round(ci["mean"]*area,2),
                "low":ci["low"],"high":ci["high"],"low_total":round(ci["low"]*area,2),"high_total":round(ci["high"]*area,2),
                "risk":ci["risk"],"risk_color":ci["risk_color"],"cv":ci["cv"],
                "alerts":smart_alerts(season,state,rainfall,area,fertilizer,crop,pesticide=pesticide)}
            if session.get("user_id"):
                conn=get_db()
                conn.execute("""INSERT INTO predictions
                    (user_id,crop,season,state,area,rainfall,fertilizer,pesticide,predicted_yield,predicted_low,predicted_high,risk_level)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (session["user_id"],crop,season,state,area,rainfall,fertilizer,pesticide,ci["mean"],ci["low"],ci["high"],ci["risk"]))
                conn.commit(); conn.close()
        except Exception as e: flash(f"Error: {e}","danger")
    return render_template("predict.html", result=result, form_data=form_data)

@app.route("/economic", methods=["GET","POST"])
def economic():
    result=None; form_data={}
    if request.method=="POST":
        try:
            crop      = request.form["crop"]
            area      = float(request.form["area"])
            exp_yield = float(request.form["exp_yield"])
            mkt_price = float(request.form.get("mkt_price") or MARKET_PRICES.get(crop, 2000))

            raw = lambda key: request.form.get(key, "").strip()
            seed_cost   = float(raw("seed_cost"))   if raw("seed_cost")   else 0
            fert_cost   = float(raw("fert_cost"))   if raw("fert_cost")   else 0
            pest_cost   = float(raw("pest_cost"))   if raw("pest_cost")   else 0
            labour_days = float(raw("labour_days")) if raw("labour_days") else 0
            labour_wage = float(raw("labour_wage")) if raw("labour_wage") else 0
            equip_cost  = float(raw("equip_cost"))  if raw("equip_cost")  else 0
            other_cost  = float(raw("other_cost"))  if raw("other_cost")  else 0

            labour_cost = round(labour_days * labour_wage)
            total_cost  = seed_cost + fert_cost + pest_cost + labour_cost + equip_cost + other_cost
            total_prod  = exp_yield * area
            revenue     = round(total_prod * mkt_price * 10)
            profit      = round(revenue - total_cost)
            roi         = round((profit / total_cost * 100), 1) if total_cost > 0 else 0
            break_even  = round(total_cost / (total_prod * 10)) if total_prod > 0 else 0

            form_data = dict(crop=crop, area=area, exp_yield=exp_yield, mkt_price=mkt_price,
                seed_cost=round(seed_cost), fert_cost=round(fert_cost), pest_cost=round(pest_cost),
                labour_days=round(labour_days), labour_wage=round(labour_wage),
                equip_cost=round(equip_cost), other_cost=round(other_cost))

            cost_items = {"Seeds":round(seed_cost),"Fertilizer":round(fert_cost),
                          "Labour":labour_cost,"Equipment Rental":round(equip_cost)}
            biggest_cost = max(cost_items, key=cost_items.get)
            biggest_amt  = cost_items[biggest_cost]

            result = dict(
                crop=crop, area=area,
                total_production_tons=round(total_prod, 2),
                revenue=revenue,
                seed_cost=round(seed_cost), fert_cost=round(fert_cost),
                pest_cost=round(pest_cost), labour_cost=labour_cost,
                labour_days=round(labour_days), labour_wage=round(labour_wage),
                equip_cost=round(equip_cost), other_cost=round(other_cost),
                total_cost=round(total_cost), profit=profit, roi=roi,
                break_even_price=break_even, mkt_price=mkt_price,
                is_profitable=profit > 0,
                biggest_cost=biggest_cost, biggest_amt=biggest_amt
            )
        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template("economic.html", result=result, form_data=form_data)

@app.route("/economic/calculate", methods=["POST"])
def economic_calculate():
    return redirect(url_for("economic"))

@app.route("/alerts", methods=["GET","POST"])
def alerts():
    result_alerts=[]; form_data={}
    if request.method=="POST":
        try:
            season=request.form["season"]; state=request.form["state"]
            rainfall=float(request.form["rainfall"]); area=float(request.form["area"])
            fertilizer=float(request.form["fertilizer"]); pesticide=float(request.form.get("pesticide",0))
            form_data=dict(season=season,state=state,rainfall=rainfall,area=area,fertilizer=fertilizer,pesticide=pesticide)
            result_alerts=smart_alerts(season,state,rainfall,area,fertilizer,pesticide=pesticide)
            if rainfall<300 and season in ["Kharif","Summer"]:
                result_alerts.append({"type":"danger","icon":"🏜️","title":"Drought Risk","msg":"Critical rainfall shortage — irrigation is mandatory for viable yield."})
        except Exception as e: flash(f"Error: {e}","danger")
    return render_template("alerts.html", alerts=result_alerts, form_data=form_data)

@app.route("/equipment")
def equipment():
    sf=request.args.get("state",""); df=request.args.get("district",""); cf=request.args.get("category","")
    conn=get_db()
    q="""SELECT e.*,u.name as owner_name,u.phone as owner_phone,
        COALESCE(AVG(r.rating),0) as avg_rating,COUNT(r.id) as review_count
        FROM equipment e JOIN users u ON e.owner_id=u.id
        LEFT JOIN ratings r ON r.equipment_id=e.id WHERE 1=1"""
    params=[]
    if sf: q+=" AND e.state=?"; params.append(sf)
    if df: q+=" AND e.district LIKE ?"; params.append(f"%{df}%")
    if cf: q+=" AND e.category=?"; params.append(cf)
    q+=" GROUP BY e.id"
    equips=conn.execute(q,params).fetchall()
    categories=[r["category"] for r in conn.execute("SELECT DISTINCT category FROM equipment").fetchall()]
    conn.close()
    return render_template("equipment.html", equipment=equips, categories=categories,
                           state_filter=sf, district_filter=df, category_filter=cf)

@app.route("/equipment_recommend", methods=["GET","POST"])
def equipment_recommend():
    recommended=[]; form_data={}
    if request.method=="POST":
        crop=request.form["crop"]; area=float(request.form.get("area",1)); state=request.form.get("state","")
        form_data=dict(crop=crop,area=area,state=state)
        rec_cats=EQUIPMENT_RULES.get(crop,["Tractor","Power Sprayer","Seed Drill"])
        conn=get_db()
        for cat in rec_cats:
            q="SELECT e.*,u.name as owner_name,COALESCE(AVG(r.rating),0) as avg_rating FROM equipment e JOIN users u ON e.owner_id=u.id LEFT JOIN ratings r ON r.equipment_id=e.id WHERE e.category LIKE ? AND e.is_available=1"
            params=[f"%{cat.split()[0]}%"]
            if state: q+=" AND e.state=?"; params.append(state)
            q+=" GROUP BY e.id LIMIT 3"
            items=conn.execute(q,params).fetchall()
            if items: recommended.append({"category":cat,"equipment_list":items})
        conn.close()
    return render_template("equipment_recommend.html", recommended=recommended, form_data=form_data)

@app.route("/equipment_calendar/<int:equip_id>")
def equipment_calendar(equip_id):
    conn=get_db()
    bookings=conn.execute("""SELECT start_date,end_date FROM bookings
        WHERE equipment_id=? AND status NOT IN ('rejected','cancelled') AND end_date >= date('now')""",(equip_id,)).fetchall()
    conn.close()
    booked=[]
    for b in bookings:
        try:
            start=datetime.strptime(b["start_date"],"%Y-%m-%d")
            end=datetime.strptime(b["end_date"],"%Y-%m-%d")
            delta=(end-start).days
            for i in range(delta+1):
                from datetime import timedelta
                booked.append((start+timedelta(days=i)).strftime("%Y-%m-%d"))
        except: pass
    return jsonify({"booked":booked})

@app.route("/book/<int:equip_id>", methods=["GET","POST"])
def book_equipment(equip_id):
    user=current_user()
    if not user: flash("Please login first.","warning"); return redirect(url_for("login"))
    conn=get_db()
    equip=conn.execute("SELECT e.*,u.name as owner,u.phone as owner_phone FROM equipment e JOIN users u ON e.owner_id=u.id WHERE e.id=?",(equip_id,)).fetchone()
    if not equip: flash("Not found.","danger"); return redirect(url_for("equipment"))
    if request.method=="POST":
        start=request.form["start_date"]; end=request.form["end_date"]
        try:
            d1=datetime.strptime(start,"%Y-%m-%d"); d2=datetime.strptime(end,"%Y-%m-%d")
            days=(d2-d1).days
            if days<=0: flash("End date must be after start date.","danger")
            else:
                conflict=conn.execute("SELECT id FROM bookings WHERE equipment_id=? AND status NOT IN ('rejected','cancelled') AND NOT (end_date<=? OR start_date>=?)",(equip_id,start,end)).fetchone()
                if conflict: flash("Already booked for these dates.","danger")
                else:
                    total=days*equip["daily_rate"]
                    conn.execute("INSERT INTO bookings (farmer_id,equipment_id,start_date,end_date,total_cost,notes,pickup_address,pickup_district,pickup_state) VALUES (?,?,?,?,?,?,?,?,?)",
                        (user["id"],equip_id,start,end,total,request.form.get("notes",""),
                         request.form.get("pickup_address",""),request.form.get("pickup_district",""),request.form.get("pickup_state","")))
                    conn.commit(); flash(f"Booking submitted! Cost: ₹{total:,.0f}","success")
                    return redirect(url_for("my_bookings"))
        except ValueError: flash("Invalid date.","danger")
    conn.close()
    return render_template("book.html", equip=equip)

@app.route("/my_bookings")
def my_bookings():
    user=current_user()
    if not user: return redirect(url_for("login"))
    conn=get_db()
    bookings=conn.execute("""SELECT b.*,e.name as equip_name,e.category,e.daily_rate,u.name as owner_name,r.rating,r.review
        FROM bookings b JOIN equipment e ON b.equipment_id=e.id JOIN users u ON e.owner_id=u.id
        LEFT JOIN ratings r ON r.booking_id=b.id WHERE b.farmer_id=? ORDER BY b.created_at DESC""",(user["id"],)).fetchall()
    conn.close()
    return render_template("my_bookings.html", bookings=bookings)

@app.route("/cancel_booking/<int:booking_id>")
def cancel_booking(booking_id):
    user=current_user()
    if not user: return redirect(url_for("login"))
    conn=get_db(); conn.execute("UPDATE bookings SET status='cancelled' WHERE id=? AND farmer_id=?",(booking_id,user["id"])); conn.commit(); conn.close()
    flash("Booking cancelled.","info"); return redirect(url_for("my_bookings"))

@app.route("/rate_booking/<int:booking_id>", methods=["GET","POST"])
def rate_booking(booking_id):
    user=current_user()
    if not user: return redirect(url_for("login"))
    conn=get_db()
    booking=conn.execute("""SELECT b.*,e.name as equip_name,e.owner_id,u.name as owner_name
        FROM bookings b JOIN equipment e ON b.equipment_id=e.id JOIN users u ON e.owner_id=u.id
        WHERE b.id=? AND b.farmer_id=?""",(booking_id,user["id"])).fetchone()
    if not booking or booking["status"]!="approved":
        flash("Cannot rate this booking.","danger"); conn.close(); return redirect(url_for("my_bookings"))
    if request.method=="POST":
        rating=int(request.form["rating"]); review=request.form.get("review","")
        try:
            conn.execute("INSERT INTO ratings (booking_id,farmer_id,equipment_id,provider_id,rating,review) VALUES (?,?,?,?,?,?)",
                (booking_id,user["id"],booking["equipment_id"],booking["owner_id"],rating,review))
            conn.commit(); flash("Rating submitted!","success")
        except: flash("Already rated.","info")
        conn.close(); return redirect(url_for("my_bookings"))
    conn.close()
    return render_template("rate_booking.html", booking=booking)

@app.route("/profile")
def profile():
    user=current_user()
    if not user: return redirect(url_for("login"))
    conn=get_db()
    preds=conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 10",(user["id"],)).fetchall()
    bookings=conn.execute("SELECT b.*,e.name as equip_name FROM bookings b JOIN equipment e ON b.equipment_id=e.id WHERE b.farmer_id=? ORDER BY b.created_at DESC",(user["id"],)).fetchall()
    total_area=sum(p["area"] for p in preds) if preds else 0
    crop_counts={}
    for p in preds: crop_counts[p["crop"]]=crop_counts.get(p["crop"],0)+1
    best_crop=max(crop_counts,key=crop_counts.get) if crop_counts else "—"
    rental_spend=sum(b["total_cost"] for b in bookings if b["status"]=="approved")
    stats={"total_preds":len(preds),"total_area":round(total_area,1),"best_crop":best_crop,"rental_spend":round(rental_spend)}
    conn.close()
    return render_template("profile.html", user=user, preds=preds, bookings=bookings, stats=stats)

@app.route("/dashboard")
def dashboard():
    user=current_user()
    if not user or user["role"] not in ("provider","admin"): flash("Provider access only.","warning"); return redirect(url_for("index"))
    conn=get_db()
    my_equip=conn.execute("SELECT e.*,COALESCE(AVG(r.rating),0) as avg_rating,COUNT(r.id) as review_count FROM equipment e LEFT JOIN ratings r ON r.equipment_id=e.id WHERE e.owner_id=? GROUP BY e.id",(user["id"],)).fetchall()
    bookings=conn.execute("""SELECT b.*,e.name as equip_name,u.name as farmer_name,u.state as farmer_state,u.phone as farmer_phone
        FROM bookings b JOIN equipment e ON b.equipment_id=e.id JOIN users u ON b.farmer_id=u.id WHERE e.owner_id=? ORDER BY b.created_at DESC""",(user["id"],)).fetchall()
    stats={"total_equip":len(my_equip),"available":sum(1 for e in my_equip if e["is_available"]),
           "pending":sum(1 for b in bookings if b["status"]=="pending"),
           "active":sum(1 for b in bookings if b["status"]=="approved"),
           "total_revenue":sum(b["total_cost"] for b in bookings if b["status"]=="approved")}
    conn.close()
    return render_template("dashboard.html", my_equip=my_equip, bookings=bookings, stats=stats)

@app.route("/booking_action/<int:booking_id>/<action>")
def booking_action(booking_id, action):
    user=current_user()
    if not user: return redirect(url_for("login"))
    conn=get_db(); conn.execute("UPDATE bookings SET status=? WHERE id=?",(action,booking_id)); conn.commit(); conn.close()
    flash(f"Booking {action}.","success"); return redirect(url_for("dashboard"))

@app.route("/add_equipment", methods=["GET","POST"])
def add_equipment():
    user=current_user()
    if not user or user["role"] not in ("provider","admin"): return redirect(url_for("login"))
    if request.method=="POST":
        conn=get_db()
        # ── Handle photo file upload ──────────────────────────────────────────
        photo_url = None
        photo_file = request.files.get("photo")
        if photo_file and photo_file.filename:
            import uuid
            from werkzeug.utils import secure_filename
            allowed = {"jpg","jpeg","png","webp"}
            ext = photo_file.filename.rsplit(".",1)[-1].lower()
            if ext in allowed:
                upload_dir = os.path.join(BASE_DIR, "static", "uploads", "equipment")
                os.makedirs(upload_dir, exist_ok=True)
                filename = f"{uuid.uuid4().hex}.{ext}"
                photo_file.save(os.path.join(upload_dir, filename))
                photo_url = f"/static/uploads/equipment/{filename}"
        conn.execute("INSERT INTO equipment (owner_id,name,category,description,daily_rate,state,district,photo_url) VALUES (?,?,?,?,?,?,?,?)",
            (user["id"],request.form["name"],request.form["category"],request.form["description"],float(request.form["daily_rate"]),request.form["state"],request.form["district"],photo_url))
        conn.commit(); conn.close(); flash("Equipment added!","success"); return redirect(url_for("dashboard"))
    return render_template("add_equipment.html", states=META["states"])

@app.route("/delete_equipment/<int:equip_id>", methods=["POST"])
def delete_equipment(equip_id):
    user = current_user()
    if not user or user["role"] not in ("provider","admin"):
        return redirect(url_for("login"))
    conn = get_db()
    # Only owner or admin can delete
    equip = conn.execute("SELECT * FROM equipment WHERE id=?", (equip_id,)).fetchone()
    if not equip or (equip["owner_id"] != user["id"] and user["role"] != "admin"):
        flash("Not authorised.", "danger"); conn.close(); return redirect(url_for("dashboard"))
    # Delete uploaded photo file from disk if it exists
    if equip["photo_url"] and equip["photo_url"].startswith("/static/uploads/"):
        try:
            file_path = os.path.join(BASE_DIR, equip["photo_url"].lstrip("/"))
            if os.path.exists(file_path):
                os.remove(file_path)
        except: pass
    # Cancel pending bookings for this equipment
    conn.execute("UPDATE bookings SET status='cancelled' WHERE equipment_id=? AND status='pending'", (equip_id,))
    conn.execute("DELETE FROM equipment WHERE id=?", (equip_id,))
    conn.commit(); conn.close()
    flash("Equipment deleted successfully.", "success")
    return redirect(url_for("dashboard"))

@app.route("/toggle_availability/<int:equip_id>")
def toggle_availability(equip_id):
    user = current_user()
    if not user or user["role"] not in ("provider","admin"):
        return redirect(url_for("login"))
    conn = get_db()
    equip = conn.execute("SELECT * FROM equipment WHERE id=?", (equip_id,)).fetchone()
    if not equip or (equip["owner_id"] != user["id"] and user["role"] != "admin"):
        flash("Not authorised.", "danger"); conn.close(); return redirect(url_for("dashboard"))
    new_val = 0 if equip["is_available"] else 1
    conn.execute("UPDATE equipment SET is_available=? WHERE id=?", (new_val, equip_id))
    conn.commit(); conn.close()
    flash(f"Equipment marked as {'Available' if new_val else 'Unavailable'}.", "success")
    return redirect(url_for("dashboard"))

@app.route("/admin")
def admin():
    user=current_user()
    if not user or user["role"]!="admin": flash("Admin only.","danger"); return redirect(url_for("index"))
    conn=get_db()
    users=conn.execute("SELECT u.*,COUNT(p.id) as pred_count FROM users u LEFT JOIN predictions p ON p.user_id=u.id GROUP BY u.id ORDER BY u.created_at DESC").fetchall()
    all_equipment=conn.execute("SELECT e.*,u.name as owner_name,COALESCE(AVG(r.rating),0) as avg_rating FROM equipment e JOIN users u ON e.owner_id=u.id LEFT JOIN ratings r ON r.equipment_id=e.id GROUP BY e.id").fetchall()
    all_bookings=conn.execute("""SELECT b.*,e.name as equip_name,uf.name as farmer_name,up.name as provider_name
        FROM bookings b JOIN equipment e ON b.equipment_id=e.id JOIN users uf ON b.farmer_id=uf.id JOIN users up ON e.owner_id=up.id
        ORDER BY b.created_at DESC LIMIT 50""").fetchall()
    acc = None
    stats={
        "total_users":conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
        "farmers":conn.execute("SELECT COUNT(*) FROM users WHERE role='farmer'").fetchone()[0],
        "providers":conn.execute("SELECT COUNT(*) FROM users WHERE role='provider'").fetchone()[0],
        "total_equipment":conn.execute("SELECT COUNT(*) FROM equipment").fetchone()[0],
        "total_bookings":conn.execute("SELECT COUNT(*) FROM bookings").fetchone()[0],
        "approved_bookings":conn.execute("SELECT COUNT(*) FROM bookings WHERE status='approved'").fetchone()[0],
        "total_predictions":conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
        "avg_accuracy":round(100-(acc or 0),1),
        "total_revenue":conn.execute("SELECT SUM(total_cost) FROM bookings WHERE status='approved'").fetchone()[0] or 0,
    }
    conn.close()
    return render_template("admin.html", users=users, all_equipment=all_equipment,
                           all_bookings=all_bookings, stats=stats)

@app.route("/admin/toggle_user/<int:uid>")
def admin_toggle_user(uid):
    user=current_user()
    if not user or user["role"]!="admin": return redirect(url_for("index"))
    conn=get_db()
    u=conn.execute("SELECT role FROM users WHERE id=?",(uid,)).fetchone()
    new_role="banned" if u and u["role"] not in ("banned","admin") else "farmer"
    conn.execute("UPDATE users SET role=? WHERE id=?",(new_role,uid)); conn.commit(); conn.close()

    flash(f"User updated to {new_role}.","success"); return redirect(url_for("admin"))

@app.route("/admin/export_csv")
def export_csv():
    user=current_user()
    if not user or user["role"]!="admin": return redirect(url_for("index"))
    conn=get_db()
    bookings=conn.execute("""SELECT b.id,uf.name as farmer,up.name as provider,e.name as equipment,
        b.start_date,b.end_date,b.total_cost,b.status,b.created_at FROM bookings b
        JOIN equipment e ON b.equipment_id=e.id JOIN users uf ON b.farmer_id=uf.id JOIN users up ON e.owner_id=up.id""").fetchall()
    conn.close()
    out=io.StringIO(); w=csv.writer(out)
    w.writerow(["ID","Farmer","Provider","Equipment","Start","End","Cost","Status","Created"])
    for b in bookings: w.writerow([b["id"],b["farmer"],b["provider"],b["equipment"],b["start_date"],b["end_date"],b["total_cost"],b["status"],b["created_at"]])
    return Response(out.getvalue(),mimetype="text/csv",headers={"Content-Disposition":"attachment;filename=bookings_export.csv"})

@app.route("/api/market_price/<crop>")
def api_market_price(crop):
    return jsonify({"price":MARKET_PRICES.get(crop,2000),"cost_per_ha":COST_PER_HA.get(crop,30000)})

@app.route("/api/reviews/<int:equip_id>")
def api_reviews(equip_id):
    conn = get_db()
    rows = conn.execute("""
        SELECT r.rating, r.review, r.created_at, u.name as farmer_name
        FROM ratings r
        JOIN users u ON r.farmer_id = u.id
        WHERE r.equipment_id = ?
        ORDER BY r.created_at DESC
    """, (equip_id,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

if __name__=="__main__":
    app.run(debug=True, port=5000)