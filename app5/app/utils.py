from flask_jwt_extended import verify_jwt_in_request

def is_user_authenticated():
    try:
        verify_jwt_in_request()
        return True
    except Exception:
        return False
