from app import create_app, db
from flask_migrate import Migrate

app = create_app()
migrate = Migrate(app, db)  # Optional: For migrations

if __name__ == "__main__":
    with app.app_context():
        from app import db
        db.create_all()
    app.run(debug=True, port=5001)
