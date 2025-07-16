from datetime import datetime
from app import db

class Transaction(db.Model):
    __tablename__ = 'transactions'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    description = db.Column(db.String(255))
    category = db.Column(db.String(50), index=True)
    amount = db.Column(db.Float, nullable=False)

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)

    def __repr__(self):
        return f'<Transaction {self.category} ${self.amount:.2f} on {self.date.date()}>'
