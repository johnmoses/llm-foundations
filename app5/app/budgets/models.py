from datetime import datetime
from app import db

class Budget(db.Model):
    __tablename__ = 'budgets'

    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False, index=True)
    limit = db.Column(db.Float, nullable=False)
    period_start = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    period_end = db.Column(db.DateTime, nullable=True)

    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    user = db.relationship('User', back_populates='budgets')

    def __repr__(self):
        period_end_str = self.period_end.strftime('%Y-%m-%d') if self.period_end else 'N/A'
        return (f'<Budget category={self.category} limit=${self.limit:.2f} '
                f'from {self.period_start.strftime("%Y-%m-%d")} to {period_end_str}>')
