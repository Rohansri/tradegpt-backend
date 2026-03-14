from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from uuid import uuid4
from datetime import datetime, date
import uvicorn
import os

app = FastAPI(title="1mg Clone API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Mock Data ───────────────────────────

MEDICINES = [
    {"id": "m001", "name": "Dolo 650 Tablet", "manufacturer": "Micro Labs Ltd", "price": 30.0, "mrp": 35.0, "composition": "Paracetamol 650mg", "category": "Pain Relief", "rating": 4.5, "reviews": 12480, "prescription_required": False, "pack_size": "15 Tablets", "description": "Used for fever and mild to moderate pain relief.", "image": "💊", "in_stock": True, "discount": 14},
    {"id": "m002", "name": "Combiflam Tablet", "manufacturer": "Sanofi India Ltd", "price": 28.0, "mrp": 32.0, "composition": "Ibuprofen 400mg + Paracetamol 325mg", "category": "Pain Relief", "rating": 4.3, "reviews": 8921, "prescription_required": False, "pack_size": "20 Tablets", "description": "Relieves pain, inflammation and fever.", "image": "💊", "in_stock": True, "discount": 12},
    {"id": "m003", "name": "Augmentin 625 Duo Tablet", "manufacturer": "GlaxoSmithKline", "price": 194.0, "mrp": 220.0, "composition": "Amoxycillin 500mg + Clavulanic Acid 125mg", "category": "Antibiotics", "rating": 4.6, "reviews": 5432, "prescription_required": True, "pack_size": "10 Tablets", "description": "Antibiotic for bacterial infections.", "image": "💊", "in_stock": True, "discount": 12},
    {"id": "m004", "name": "Crocin Advance Tablet", "manufacturer": "GlaxoSmithKline", "price": 30.0, "mrp": 34.0, "composition": "Paracetamol 500mg", "category": "Pain Relief", "rating": 4.4, "reviews": 9823, "prescription_required": False, "pack_size": "20 Tablets", "description": "Fast-acting pain and fever relief.", "image": "💊", "in_stock": True, "discount": 12},
    {"id": "m005", "name": "Allegra 120mg Tablet", "manufacturer": "Sanofi India Ltd", "price": 127.0, "mrp": 145.0, "composition": "Fexofenadine 120mg", "category": "Allergy", "rating": 4.5, "reviews": 6234, "prescription_required": False, "pack_size": "10 Tablets", "description": "Non-drowsy antihistamine for allergies.", "image": "💊", "in_stock": True, "discount": 12},
    {"id": "m006", "name": "Metformin 500mg Tablet", "manufacturer": "Sun Pharma", "price": 22.0, "mrp": 28.0, "composition": "Metformin 500mg", "category": "Diabetes", "rating": 4.4, "reviews": 7654, "prescription_required": True, "pack_size": "20 Tablets", "description": "Used to manage blood sugar in type 2 diabetes.", "image": "💊", "in_stock": True, "discount": 21},
    {"id": "m007", "name": "Pantocid 40mg Tablet", "manufacturer": "Sun Pharma", "price": 99.0, "mrp": 115.0, "composition": "Pantoprazole 40mg", "category": "Gastro", "rating": 4.3, "reviews": 4321, "prescription_required": False, "pack_size": "15 Tablets", "description": "Reduces stomach acid for acidity and GERD.", "image": "💊", "in_stock": True, "discount": 14},
    {"id": "m008", "name": "Azithral 500 Tablet", "manufacturer": "Alembic Pharma", "price": 85.0, "mrp": 98.0, "composition": "Azithromycin 500mg", "category": "Antibiotics", "rating": 4.5, "reviews": 3214, "prescription_required": True, "pack_size": "3 Tablets", "description": "Broad-spectrum antibiotic.", "image": "💊", "in_stock": True, "discount": 13},
    {"id": "m009", "name": "Sinarest Tablet", "manufacturer": "Centaur Pharma", "price": 15.0, "mrp": 18.0, "composition": "Paracetamol + Chlorpheniramine + Phenylephrine", "category": "Cold & Cough", "rating": 4.2, "reviews": 8762, "prescription_required": False, "pack_size": "10 Tablets", "description": "Relieves cold, cough, and nasal congestion.", "image": "💊", "in_stock": True, "discount": 17},
    {"id": "m010", "name": "Volini Gel 30g", "manufacturer": "Sanofi India", "price": 130.0, "mrp": 148.0, "composition": "Diclofenac + Methyl Salicylate", "category": "Pain Relief", "rating": 4.5, "reviews": 11203, "prescription_required": False, "pack_size": "30g Tube", "description": "Topical pain relief gel for muscle pain.", "image": "🧴", "in_stock": True, "discount": 12},
    {"id": "m011", "name": "Telma 40 Tablet", "manufacturer": "Glenmark Pharma", "price": 118.0, "mrp": 138.0, "composition": "Telmisartan 40mg", "category": "Heart Health", "rating": 4.4, "reviews": 3421, "prescription_required": True, "pack_size": "10 Tablets", "description": "Used for hypertension (high blood pressure).", "image": "💊", "in_stock": True, "discount": 14},
    {"id": "m012", "name": "Vitamin D3 60000 IU Capsule", "manufacturer": "Mankind Pharma", "price": 89.0, "mrp": 102.0, "composition": "Cholecalciferol 60000 IU", "category": "Vitamins", "rating": 4.6, "reviews": 15234, "prescription_required": False, "pack_size": "4 Capsules", "description": "Treats vitamin D deficiency.", "image": "💊", "in_stock": True, "discount": 13},
    {"id": "m013", "name": "Becosules Capsule", "manufacturer": "Pfizer", "price": 165.0, "mrp": 190.0, "composition": "Multivitamin + Vitamin B Complex", "category": "Vitamins", "rating": 4.5, "reviews": 21456, "prescription_required": False, "pack_size": "30 Capsules", "description": "Complete vitamin and mineral supplement.", "image": "💊", "in_stock": True, "discount": 13},
    {"id": "m014", "name": "Montair LC Tablet", "manufacturer": "Cipla", "price": 143.0, "mrp": 165.0, "composition": "Montelukast 10mg + Levocetirizine 5mg", "category": "Allergy", "rating": 4.4, "reviews": 5678, "prescription_required": True, "pack_size": "10 Tablets", "description": "For allergic rhinitis and asthma.", "image": "💊", "in_stock": True, "discount": 13},
    {"id": "m015", "name": "Glycomet GP 1 Tablet", "manufacturer": "USV Ltd", "price": 56.0, "mrp": 65.0, "composition": "Glimepiride 1mg + Metformin 500mg", "category": "Diabetes", "rating": 4.3, "reviews": 2341, "prescription_required": True, "pack_size": "15 Tablets", "description": "Combination antidiabetic medicine.", "image": "💊", "in_stock": True, "discount": 14},
    {"id": "m016", "name": "Omez 20mg Capsule", "manufacturer": "Dr. Reddy's", "price": 44.0, "mrp": 52.0, "composition": "Omeprazole 20mg", "category": "Gastro", "rating": 4.4, "reviews": 7890, "prescription_required": False, "pack_size": "10 Capsules", "description": "Treats acidity, ulcers and GERD.", "image": "💊", "in_stock": True, "discount": 15},
    {"id": "m017", "name": "Atorva 10 Tablet", "manufacturer": "Zydus Cadila", "price": 72.0, "mrp": 85.0, "composition": "Atorvastatin 10mg", "category": "Heart Health", "rating": 4.4, "reviews": 4532, "prescription_required": True, "pack_size": "15 Tablets", "description": "Lowers bad cholesterol levels.", "image": "💊", "in_stock": True, "discount": 15},
    {"id": "m018", "name": "Limcee 500mg Chewable Tablet", "manufacturer": "Abbott India", "price": 30.0, "mrp": 36.0, "composition": "Vitamin C 500mg", "category": "Vitamins", "rating": 4.6, "reviews": 18234, "prescription_required": False, "pack_size": "15 Tablets", "description": "Boosts immunity with vitamin C.", "image": "💊", "in_stock": True, "discount": 17},
    {"id": "m019", "name": "Cough Syrup Benadryl 100ml", "manufacturer": "Johnson & Johnson", "price": 92.0, "mrp": 105.0, "composition": "Diphenhydramine + Ammonium Chloride", "category": "Cold & Cough", "rating": 4.2, "reviews": 9123, "prescription_required": False, "pack_size": "100ml", "description": "Relieves cough and cold symptoms.", "image": "🍶", "in_stock": True, "discount": 12},
    {"id": "m020", "name": "Shelcal 500 Tablet", "manufacturer": "Elder Pharma", "price": 98.0, "mrp": 115.0, "composition": "Calcium Carbonate 500mg + Vitamin D3", "category": "Vitamins", "rating": 4.5, "reviews": 6543, "prescription_required": False, "pack_size": "15 Tablets", "description": "Calcium supplement for bone health.", "image": "💊", "in_stock": True, "discount": 15},
]

LAB_TESTS = [
    {"id": "l001", "name": "Complete Blood Count (CBC)", "price": 299.0, "mrp": 400.0, "parameters": 26, "turnaround_time": "Same Day", "category": "Blood Test", "description": "Evaluates overall health and detects disorders like anaemia, infection.", "fasting": "No fasting required", "sample": "Blood", "discount": 25, "popular": True, "icon": "🩸"},
    {"id": "l002", "name": "Thyroid Profile (T3, T4, TSH)", "price": 499.0, "mrp": 700.0, "parameters": 3, "turnaround_time": "Same Day", "category": "Hormone Test", "description": "Detects thyroid disorders including hypothyroidism and hyperthyroidism.", "fasting": "No fasting required", "sample": "Blood", "discount": 29, "popular": True, "icon": "🔬"},
    {"id": "l003", "name": "Lipid Profile", "price": 399.0, "mrp": 550.0, "parameters": 8, "turnaround_time": "Same Day", "category": "Blood Test", "description": "Checks cholesterol and triglyceride levels for heart health.", "fasting": "10-12 hours fasting", "sample": "Blood", "discount": 27, "popular": True, "icon": "❤️"},
    {"id": "l004", "name": "Liver Function Test (LFT)", "price": 449.0, "mrp": 600.0, "parameters": 11, "turnaround_time": "Same Day", "category": "Blood Test", "description": "Assesses liver health and detects liver diseases.", "fasting": "8-10 hours fasting", "sample": "Blood", "discount": 25, "popular": False, "icon": "🔬"},
    {"id": "l005", "name": "HbA1c (Diabetes Test)", "price": 349.0, "mrp": 480.0, "parameters": 1, "turnaround_time": "Same Day", "category": "Diabetes", "description": "Measures average blood sugar over 3 months for diabetes management.", "fasting": "No fasting required", "sample": "Blood", "discount": 27, "popular": True, "icon": "🩺"},
    {"id": "l006", "name": "Vitamin D Total (25-OH)", "price": 699.0, "mrp": 950.0, "parameters": 1, "turnaround_time": "24 Hours", "category": "Vitamin Test", "description": "Checks Vitamin D levels for bone and immune health.", "fasting": "No fasting required", "sample": "Blood", "discount": 26, "popular": True, "icon": "☀️"},
    {"id": "l007", "name": "Vitamin B12", "price": 499.0, "mrp": 700.0, "parameters": 1, "turnaround_time": "24 Hours", "category": "Vitamin Test", "description": "Checks B12 levels important for nerve function and energy.", "fasting": "No fasting required", "sample": "Blood", "discount": 29, "popular": False, "icon": "🔬"},
    {"id": "l008", "name": "Kidney Function Test (KFT)", "price": 449.0, "mrp": 600.0, "parameters": 9, "turnaround_time": "Same Day", "category": "Blood Test", "description": "Evaluates kidney health and detects kidney disorders.", "fasting": "8 hours fasting preferred", "sample": "Blood & Urine", "discount": 25, "popular": False, "icon": "🫘"},
    {"id": "l009", "name": "Urine Routine Examination", "price": 149.0, "mrp": 200.0, "parameters": 24, "turnaround_time": "Same Day", "category": "Urine Test", "description": "Detects urinary tract infections and kidney disorders.", "fasting": "First morning sample preferred", "sample": "Urine", "discount": 26, "popular": False, "icon": "🧪"},
    {"id": "l010", "name": "Blood Glucose Fasting (FBS)", "price": 99.0, "mrp": 150.0, "parameters": 1, "turnaround_time": "Same Day", "category": "Diabetes", "description": "Measures fasting blood sugar levels for diabetes screening.", "fasting": "8-10 hours fasting required", "sample": "Blood", "discount": 34, "popular": True, "icon": "🩸"},
    {"id": "l011", "name": "COVID-19 RT-PCR Test", "price": 499.0, "mrp": 700.0, "parameters": 1, "turnaround_time": "24-48 Hours", "category": "Infection", "description": "Detects active COVID-19 infection.", "fasting": "No fasting required", "sample": "Nasal/Throat Swab", "discount": 29, "popular": False, "icon": "🦠"},
    {"id": "l012", "name": "Aarogyam Full Body Checkup", "price": 1999.0, "mrp": 3500.0, "parameters": 87, "turnaround_time": "48 Hours", "category": "Health Package", "description": "Comprehensive full body health check with 87 parameters.", "fasting": "10-12 hours fasting", "sample": "Blood & Urine", "discount": 43, "popular": True, "icon": "🏥"},
    {"id": "l013", "name": "Iron Studies", "price": 599.0, "mrp": 800.0, "parameters": 5, "turnaround_time": "24 Hours", "category": "Blood Test", "description": "Checks iron levels for anaemia diagnosis.", "fasting": "No fasting required", "sample": "Blood", "discount": 25, "popular": False, "icon": "🔬"},
    {"id": "l014", "name": "Dengue NS1 Antigen", "price": 799.0, "mrp": 1100.0, "parameters": 1, "turnaround_time": "Same Day", "category": "Infection", "description": "Early detection of dengue fever infection.", "fasting": "No fasting required", "sample": "Blood", "discount": 27, "popular": False, "icon": "🦟"},
    {"id": "l015", "name": "Testosterone Total", "price": 699.0, "mrp": 950.0, "parameters": 1, "turnaround_time": "24 Hours", "category": "Hormone Test", "description": "Measures testosterone levels for hormonal health.", "fasting": "Morning sample preferred", "sample": "Blood", "discount": 26, "popular": False, "icon": "🔬"},
]

DOCTORS = [
    {"id": "d001", "name": "Dr. Priya Sharma", "specialization": "General Physician", "experience": 12, "rating": 4.8, "reviews": 2341, "fee": 499.0, "availability": "Today", "next_slot": "11:30 AM", "languages": ["Hindi", "English"], "hospital": "Apollo Clinic, Delhi", "image": "👩‍⚕️", "about": "Expert in preventive care and chronic disease management. Known for patient-friendly approach.", "education": "MBBS, MD - AIIMS Delhi"},
    {"id": "d002", "name": "Dr. Rajesh Kumar", "specialization": "Cardiologist", "experience": 18, "rating": 4.9, "reviews": 1876, "fee": 999.0, "availability": "Today", "next_slot": "03:00 PM", "languages": ["Hindi", "English", "Punjabi"], "hospital": "Fortis Heart Institute, Mumbai", "image": "👨‍⚕️", "about": "Specialist in interventional cardiology and heart failure management.", "education": "MBBS, MD, DM (Cardiology) - PGI Chandigarh"},
    {"id": "d003", "name": "Dr. Ananya Reddy", "specialization": "Dermatologist", "experience": 10, "rating": 4.7, "reviews": 3120, "fee": 699.0, "availability": "Tomorrow", "next_slot": "10:00 AM", "languages": ["English", "Telugu", "Hindi"], "hospital": "Skin Care Centre, Hyderabad", "image": "👩‍⚕️", "about": "Expert in cosmetic dermatology, acne treatment, and skin conditions.", "education": "MBBS, MD (Dermatology) - Osmania Medical College"},
    {"id": "d004", "name": "Dr. Vikram Singh", "specialization": "Orthopedic Surgeon", "experience": 15, "rating": 4.8, "reviews": 1543, "fee": 899.0, "availability": "Today", "next_slot": "05:30 PM", "languages": ["Hindi", "English"], "hospital": "Max Super Specialty Hospital, Noida", "image": "👨‍⚕️", "about": "Specialized in joint replacement and sports injury treatment.", "education": "MBBS, MS (Ortho) - KGMC Lucknow"},
    {"id": "d005", "name": "Dr. Meena Iyer", "specialization": "Gynecologist", "experience": 20, "rating": 4.9, "reviews": 4231, "fee": 799.0, "availability": "Today", "next_slot": "12:00 PM", "languages": ["English", "Tamil", "Hindi"], "hospital": "Rainbow Hospital, Chennai", "image": "👩‍⚕️", "about": "Expert in high-risk pregnancy, PCOS, and minimal invasive gynecological surgeries.", "education": "MBBS, MS, DNB (OBG) - Madras Medical College"},
    {"id": "d006", "name": "Dr. Arun Patel", "specialization": "Pediatrician", "experience": 14, "rating": 4.8, "reviews": 2876, "fee": 599.0, "availability": "Today", "next_slot": "04:00 PM", "languages": ["Gujarati", "Hindi", "English"], "hospital": "Children's Hospital, Ahmedabad", "image": "👨‍⚕️", "about": "Specialist in child development, vaccination, and neonatal care.", "education": "MBBS, DCH, MD (Pediatrics) - BJ Medical College"},
    {"id": "d007", "name": "Dr. Sunita Gupta", "specialization": "Psychiatrist", "experience": 11, "rating": 4.7, "reviews": 987, "fee": 799.0, "availability": "Tomorrow", "next_slot": "11:00 AM", "languages": ["Hindi", "English"], "hospital": "Mind Wellness Clinic, Bangalore", "image": "👩‍⚕️", "about": "Specializes in anxiety disorders, depression, and cognitive behavioral therapy.", "education": "MBBS, MD (Psychiatry) - NIMHANS Bangalore"},
    {"id": "d008", "name": "Dr. Mohammed Ali", "specialization": "Endocrinologist", "experience": 16, "rating": 4.8, "reviews": 1234, "fee": 1099.0, "availability": "Today", "next_slot": "02:00 PM", "languages": ["Urdu", "Hindi", "English"], "hospital": "Aster CMI Hospital, Hyderabad", "image": "👨‍⚕️", "about": "Expert in diabetes, thyroid disorders, and hormonal imbalances.", "education": "MBBS, MD, DM (Endocrinology) - JIPMER"},
    {"id": "d009", "name": "Dr. Kavya Nair", "specialization": "Ophthalmologist", "experience": 9, "rating": 4.6, "reviews": 1876, "fee": 599.0, "availability": "Today", "next_slot": "06:00 PM", "languages": ["Malayalam", "English", "Hindi"], "hospital": "Aravind Eye Hospital, Kerala", "image": "👩‍⚕️", "about": "Specialist in cataract surgery, LASIK, and retinal disorders.", "education": "MBBS, MS (Ophthalmology) - SCTIMST"},
    {"id": "d010", "name": "Dr. Sameer Joshi", "specialization": "Neurologist", "experience": 17, "rating": 4.9, "reviews": 1098, "fee": 1199.0, "availability": "Tomorrow", "next_slot": "09:30 AM", "languages": ["Marathi", "Hindi", "English"], "hospital": "Kokilaben Hospital, Mumbai", "image": "👨‍⚕️", "about": "Expert in epilepsy, stroke management, and movement disorders.", "education": "MBBS, MD, DM (Neurology) - KEM Mumbai"},
]

CATEGORIES = [
    {"id": "pain-relief", "name": "Pain Relief", "icon": "💊", "color": "#FF6B6B"},
    {"id": "cold-cough", "name": "Cold & Cough", "icon": "🤧", "color": "#4ECDC4"},
    {"id": "vitamins", "name": "Vitamins", "icon": "⚡", "color": "#FFE66D"},
    {"id": "diabetes", "name": "Diabetes", "icon": "🩸", "color": "#A8E6CF"},
    {"id": "heart-health", "name": "Heart Health", "icon": "❤️", "color": "#FF8B94"},
    {"id": "antibiotics", "name": "Antibiotics", "icon": "🦠", "color": "#DDA0DD"},
    {"id": "gastro", "name": "Gastro Care", "icon": "🫃", "color": "#87CEEB"},
    {"id": "allergy", "name": "Allergy", "icon": "🌺", "color": "#98D8C8"},
]

# ─────────────────────────── Pydantic Models ───────────────────────────

class CartItem(BaseModel):
    medicine_id: str
    quantity: int = 1

class BookLabTest(BaseModel):
    test_id: str
    patient_name: str
    patient_age: int
    patient_gender: str
    date: str
    address: str
    phone: str

class BookDoctor(BaseModel):
    doctor_id: str
    patient_name: str
    patient_age: int
    patient_gender: str
    date: str
    slot: str
    phone: str
    symptoms: Optional[str] = ""

class PlaceOrder(BaseModel):
    session_id: str
    name: str
    phone: str
    address: str
    pincode: str
    payment_method: str = "COD"

# ─────────────────────────── In-memory Stores ───────────────────────────

carts: dict = {}         # session_id -> list of {medicine, quantity}
orders: dict = {}        # order_id -> order details
lab_bookings: dict = {}  # booking_id -> booking details
doc_bookings: dict = {}  # booking_id -> booking details

# ─────────────────────────── Root / Static ───────────────────────────

@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse("static/index.html")

# ─────────────────────────── Medicine Endpoints ───────────────────────────

@app.get("/api/medicines")
async def list_medicines(
    category: Optional[str] = None,
    page: int = 1,
    limit: int = 12,
):
    results = MEDICINES
    if category:
        results = [m for m in results if m["category"].lower() == category.lower()]
    total = len(results)
    start = (page - 1) * limit
    paginated = results[start:start + limit]
    return {"total": total, "page": page, "limit": limit, "medicines": paginated}

@app.get("/api/medicines/search")
async def search_medicines(q: str = Query(..., min_length=1)):
    q_lower = q.lower()
    results = [
        m for m in MEDICINES
        if q_lower in m["name"].lower()
        or q_lower in m["composition"].lower()
        or q_lower in m["category"].lower()
        or q_lower in m["manufacturer"].lower()
    ]
    return {"query": q, "total": len(results), "medicines": results}

@app.get("/api/medicines/{medicine_id}")
async def get_medicine(medicine_id: str):
    med = next((m for m in MEDICINES if m["id"] == medicine_id), None)
    if not med:
        raise HTTPException(status_code=404, detail="Medicine not found")
    substitutes = [m for m in MEDICINES if m["category"] == med["category"] and m["id"] != medicine_id][:3]
    return {**med, "substitutes": substitutes}

@app.get("/api/categories")
async def get_categories():
    return {"categories": CATEGORIES}

# ─────────────────────────── Cart Endpoints ───────────────────────────

@app.get("/api/cart/{session_id}")
async def get_cart(session_id: str):
    cart = carts.get(session_id, [])
    total = sum(item["medicine"]["price"] * item["quantity"] for item in cart)
    item_count = sum(item["quantity"] for item in cart)
    return {"session_id": session_id, "items": cart, "total": round(total, 2), "item_count": item_count}

@app.post("/api/cart/{session_id}/add")
async def add_to_cart(session_id: str, item: CartItem):
    med = next((m for m in MEDICINES if m["id"] == item.medicine_id), None)
    if not med:
        raise HTTPException(status_code=404, detail="Medicine not found")
    if session_id not in carts:
        carts[session_id] = []
    cart = carts[session_id]
    existing = next((c for c in cart if c["medicine"]["id"] == item.medicine_id), None)
    if existing:
        existing["quantity"] += item.quantity
    else:
        cart.append({"medicine": med, "quantity": item.quantity})
    total = sum(i["medicine"]["price"] * i["quantity"] for i in cart)
    return {"message": "Added to cart", "item_count": sum(i["quantity"] for i in cart), "total": round(total, 2)}

@app.delete("/api/cart/{session_id}/remove/{medicine_id}")
async def remove_from_cart(session_id: str, medicine_id: str):
    if session_id not in carts:
        raise HTTPException(status_code=404, detail="Cart not found")
    carts[session_id] = [c for c in carts[session_id] if c["medicine"]["id"] != medicine_id]
    total = sum(i["medicine"]["price"] * i["quantity"] for i in carts[session_id])
    return {"message": "Removed from cart", "total": round(total, 2)}

@app.put("/api/cart/{session_id}/update/{medicine_id}")
async def update_cart_quantity(session_id: str, medicine_id: str, quantity: int = Query(..., ge=1)):
    if session_id not in carts:
        raise HTTPException(status_code=404, detail="Cart not found")
    item = next((c for c in carts[session_id] if c["medicine"]["id"] == medicine_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not in cart")
    item["quantity"] = quantity
    total = sum(i["medicine"]["price"] * i["quantity"] for i in carts[session_id])
    return {"message": "Updated", "total": round(total, 2)}

# ─────────────────────────── Order Endpoints ───────────────────────────

@app.post("/api/orders")
async def place_order(order: PlaceOrder):
    cart = carts.get(order.session_id, [])
    if not cart:
        raise HTTPException(status_code=400, detail="Cart is empty")
    order_id = "ORD" + uuid4().hex[:8].upper()
    total = sum(item["medicine"]["price"] * item["quantity"] for item in cart)
    delivery_charge = 0 if total >= 499 else 49
    orders[order_id] = {
        "order_id": order_id,
        "items": cart,
        "subtotal": round(total, 2),
        "delivery_charge": delivery_charge,
        "total": round(total + delivery_charge, 2),
        "name": order.name,
        "phone": order.phone,
        "address": order.address,
        "pincode": order.pincode,
        "payment_method": order.payment_method,
        "status": "Confirmed",
        "estimated_delivery": "2-4 Business Days",
        "created_at": datetime.now().isoformat(),
    }
    carts[order.session_id] = []
    return orders[order_id]

@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    order = orders.get(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

# ─────────────────────────── Lab Test Endpoints ───────────────────────────

@app.get("/api/labs")
async def list_labs(category: Optional[str] = None, popular: Optional[bool] = None):
    results = LAB_TESTS
    if category:
        results = [t for t in results if t["category"].lower() == category.lower()]
    if popular is not None:
        results = [t for t in results if t["popular"] == popular]
    return {"total": len(results), "tests": results}

@app.get("/api/labs/search")
async def search_labs(q: str = Query(..., min_length=1)):
    q_lower = q.lower()
    results = [
        t for t in LAB_TESTS
        if q_lower in t["name"].lower()
        or q_lower in t["category"].lower()
        or q_lower in t["description"].lower()
    ]
    return {"query": q, "total": len(results), "tests": results}

@app.get("/api/labs/{test_id}")
async def get_lab_test(test_id: str):
    test = next((t for t in LAB_TESTS if t["id"] == test_id), None)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    return test

@app.post("/api/labs/book")
async def book_lab_test(booking: BookLabTest):
    test = next((t for t in LAB_TESTS if t["id"] == booking.test_id), None)
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")
    booking_id = "LAB" + uuid4().hex[:8].upper()
    lab_bookings[booking_id] = {
        "booking_id": booking_id,
        "test": test,
        "patient_name": booking.patient_name,
        "patient_age": booking.patient_age,
        "patient_gender": booking.patient_gender,
        "date": booking.date,
        "address": booking.address,
        "phone": booking.phone,
        "status": "Confirmed",
        "sample_collection": "Home Collection",
        "created_at": datetime.now().isoformat(),
    }
    return lab_bookings[booking_id]

# ─────────────────────────── Doctor Endpoints ───────────────────────────

@app.get("/api/doctors")
async def list_doctors(specialization: Optional[str] = None):
    results = DOCTORS
    if specialization:
        results = [d for d in results if specialization.lower() in d["specialization"].lower()]
    return {"total": len(results), "doctors": results}

@app.get("/api/doctors/search")
async def search_doctors(q: str = Query(..., min_length=1)):
    q_lower = q.lower()
    results = [
        d for d in DOCTORS
        if q_lower in d["name"].lower()
        or q_lower in d["specialization"].lower()
        or q_lower in d["hospital"].lower()
    ]
    return {"query": q, "total": len(results), "doctors": results}

@app.get("/api/doctors/{doctor_id}")
async def get_doctor(doctor_id: str):
    doc = next((d for d in DOCTORS if d["id"] == doctor_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return doc

@app.post("/api/doctors/book")
async def book_doctor(booking: BookDoctor):
    doc = next((d for d in DOCTORS if d["id"] == booking.doctor_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Doctor not found")
    booking_id = "DOC" + uuid4().hex[:8].upper()
    doc_bookings[booking_id] = {
        "booking_id": booking_id,
        "doctor": doc,
        "patient_name": booking.patient_name,
        "patient_age": booking.patient_age,
        "patient_gender": booking.patient_gender,
        "date": booking.date,
        "slot": booking.slot,
        "phone": booking.phone,
        "symptoms": booking.symptoms,
        "status": "Confirmed",
        "type": "Video Consultation",
        "created_at": datetime.now().isoformat(),
    }
    return doc_bookings[booking_id]

# ─────────────────────────── Entry Point ───────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
