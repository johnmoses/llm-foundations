from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.budgets.models import Budget
from datetime import datetime

budgets_bp = Blueprint('budgets_bp', __name__, template_folder='../templates/budgets')

@budgets_bp.route('/')
@jwt_required()
def budget_overview():
    user_id = get_jwt_identity()
    budgets = Budget.query.filter_by(user_id=user_id).order_by(Budget.category).all()
    return render_template('budgets/overview.html', budgets=budgets, is_authenticated=True)

@budgets_bp.route('/add', methods=['GET', 'POST'])
@jwt_required()
def add_budget():
    if request.method == 'POST':
        user_id = get_jwt_identity()
        category = request.form.get('category')
        limit = request.form.get('limit')
        period_start = request.form.get('period_start')
        period_end = request.form.get('period_end')

        if not category or not limit or not period_start:
            flash("Category, Limit and Period Start are required.", "error")
            return redirect(url_for('budgets_bp.add_budget'))

        try:
            limit_val = float(limit)
            period_start_dt = datetime.strptime(period_start, '%Y-%m-%d')
            period_end_dt = datetime.strptime(period_end, '%Y-%m-%d') if period_end else None
        except ValueError:
            flash("Invalid input format.", "error")
            return redirect(url_for('budgets_bp.add_budget'))

        new_budget = Budget(
            category=category,
            limit=limit_val,
            period_start=period_start_dt,
            period_end=period_end_dt,
            user_id=user_id
        )
        db.session.add(new_budget)
        db.session.commit()
        flash("Budget added successfully.", "success")
        return redirect(url_for('budgets_bp.budget_overview'))

    return render_template('budgets/add_budget.html', budget=None)


@budgets_bp.route('/edit/<int:budget_id>', methods=['GET', 'POST'])
@jwt_required()
def edit_budget(budget_id):
    user_id = get_jwt_identity()
    budget = Budget.query.filter_by(id=budget_id, user_id=user_id).first_or_404()

    if request.method == 'POST':
        category = request.form.get('category')
        limit = request.form.get('limit')
        period_start = request.form.get('period_start')
        period_end = request.form.get('period_end')

        if not category or not limit or not period_start:
            flash("Category, Limit and Period Start are required.", "error")
            return redirect(url_for('budgets_bp.edit_budget', budget_id=budget_id))

        try:
            budget.category = category
            budget.limit = float(limit)
            budget.period_start = datetime.strptime(period_start, '%Y-%m-%d')
            budget.period_end = datetime.strptime(period_end, '%Y-%m-%d') if period_end else None
        except ValueError:
            flash("Invalid input format.", "error")
            return redirect(url_for('budgets_bp.edit_budget', budget_id=budget_id))

        db.session.commit()
        flash("Budget updated successfully.", "success")
        return redirect(url_for('budgets_bp.budget_overview'))

    return render_template('budgets/add.html', budget=budget)

@budgets_bp.route('/delete/<int:budget_id>', methods=['POST'])
@jwt_required()
def delete_budget(budget_id):
    user_id = get_jwt_identity()
    budget = Budget.query.filter_by(id=budget_id, user_id=user_id).first_or_404()
    db.session.delete(budget)
    db.session.commit()
    flash("Budget deleted.", "success")
    return redirect(url_for('budgets_bp.budget_overview'))

@budgets_bp.route('/api/budgets')
@jwt_required()
def budgets_api():
    user_id = get_jwt_identity()
    budgets = Budget.query.filter_by(user_id=user_id).order_by(Budget.category).all()

    # Assuming Budget model has serialize method converting model to dict
    serialized = [b.serialize() for b in budgets]  
    return jsonify(serialized)