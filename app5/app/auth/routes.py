from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from app import db
from app.auth.models import User
from flask_jwt_extended import create_access_token, set_access_cookies, unset_jwt_cookies

auth_bp = Blueprint('auth_bp', __name__, template_folder='../templates/auth')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Try to get json data silently, otherwise use form data.
        data = request.get_json(silent=True)
        if data is None:
            data = request.form
        
        # Validate data keys exist (basic)
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"msg": "Missing username or password"}), 400

        if User.query.filter_by(username=data['username']).first():
            return jsonify({"msg": "Username already exists"}), 400

        user = User(username=data['username'], email=data.get('email'))
        user.set_password(data['password'])
        db.session.add(user)
        db.session.commit()
        # flash("User created successfully. Please log in.")
        return redirect(url_for('auth_bp.login'))

    return render_template('auth/signup.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"msg": "Missing username or password"}), 400
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            access_token = create_access_token(identity=user.id)
            resp = jsonify({'msg': 'Login successful'})
            set_access_cookies(resp, access_token)  # Set cookie with token + CSRF cookie
            return resp, 200
        else:
            return jsonify({"msg": "Invalid username or password"}), 401
    
    # For GET or other methods, render login template normally
    return render_template('auth/login.html')

@auth_bp.route('/logout', methods=['GET'])
def logout():
    resp = jsonify({"msg": "Logout successful"})
    unset_jwt_cookies(resp)  # Clear JWT and CSRF cookies
    return render_template('auth/login.html')