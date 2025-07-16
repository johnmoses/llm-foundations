from flask import Flask, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from datetime import timedelta
from .utils import is_user_authenticated

db = SQLAlchemy()
jwt = JWTManager()


def create_app(config_class=None):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(config_class or "config.Config")

    app.config["JWT_SECRET_KEY"] = "your-secret-key"  # Use a secure secret!
    app.config["JWT_TOKEN_LOCATION"] = ["cookies"]  # Store JWT in cookies
    app.config["JWT_ACCESS_COOKIE_PATH"] = "/"  # Cookie available on all routes
    app.config["JWT_COOKIE_SECURE"] = False  # True if serving HTTPS in production
    app.config["JWT_COOKIE_CSRF_PROTECT"] = (
        False  # Can enable for CSRF protection (optional)
    )
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(minutes=60)
    # app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)  # refresh token lasts longer
    # Keep JWT_ACCESS_TOKEN_EXPIRES shorter, e.g. 15 minutes (default)

    jwt = JWTManager(app)

    db.init_app(app)
    jwt.init_app(app)

    # Import and register blueprints
    from app.auth.routes import auth_bp
    from app.transactions.routes import transactions_bp
    from app.budgets.routes import budgets_bp
    from app.chat.routes import chat_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(transactions_bp, url_prefix="/transactions")
    app.register_blueprint(budgets_bp, url_prefix="/budgets")
    app.register_blueprint(chat_bp, url_prefix="/chat")

    @app.route("/")
    def root():
        return redirect(url_for("auth_bp.login"))

    @app.context_processor
    def inject_auth_status():
        return dict(is_authenticated=is_user_authenticated())

    return app
