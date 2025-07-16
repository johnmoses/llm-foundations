import random
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, render_template
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.transactions.models import Transaction

transactions_bp = Blueprint('transactions_bp', __name__, template_folder='../templates/transactions')

CATEGORIES = ['Groceries', 'Entertainment', 'Transport', 'Utilities', 'Dining', 'Health']
MERCHANTS = {
    'Groceries': ['SuperMart', 'FreshFarm', 'GroceryHub'],
    'Entertainment': ['MoviePlex', 'StreamingPlus', 'ArcadeWorld'],
    'Transport': ['CityRail', 'UberRide', 'FuelStation'],
    'Utilities': ['PowerGrid', 'WaterCo', 'InternetTel'],
    'Dining': ['PizzaPlace', 'SushiCorner', 'BurgerBarn'],
    'Health': ['PharmaPlus', 'HealthClinic', 'DentistCare']
}

@transactions_bp.route('/')
@jwt_required()
def dashboard():
    user_id = get_jwt_identity()
    transactions = Transaction.query.filter_by(user_id=user_id).order_by(Transaction.date.desc()).all()
    return render_template('transactions/dashboard.html', transactions=transactions)

@transactions_bp.route('/add', methods=['GET', 'POST'])
@jwt_required()
def add_transaction():
    if request.method == 'POST':
        data = request.json or request.form
        tx = Transaction(
            date=datetime.strptime(data.get('date'), '%Y-%m-%d %H:%M:%S') if data.get('date') else datetime.utcnow(),
            description=data.get('description'),
            category=data.get('category'),
            amount=float(data.get('amount')),
            user_id=get_jwt_identity()
        )
        db.session.add(tx)
        db.session.commit()
        return jsonify({"msg": "Transaction added"}), 201

    return render_template('transactions/add_transaction.html', categories=CATEGORIES)

@transactions_bp.route('/generate-demo', methods=['POST'])
@jwt_required()
def generate_demo_transactions():
    user_id = get_jwt_identity()
    demo_txs = []
    for _ in range(5):
        category = random.choice(CATEGORIES)
        merchant = random.choice(MERCHANTS[category])
        amount = round(random.uniform(5, 200), 2)
        timestamp = datetime.now() - timedelta(minutes=random.randint(0, 60))
        description = f"{merchant} - {category}"
        tx = Transaction(
            date=timestamp,
            description=description,
            category=category,
            amount=amount,
            user_id=user_id
        )
        db.session.add(tx)
        demo_txs.append({
            "date": timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            "description": description,
            "category": category,
            "amount": amount
        })
    db.session.commit()
    return jsonify(demo_txs)
