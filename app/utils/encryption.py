"""
Encryption utils
"""

from cryptography.fernet import Fernet
from flask import current_app

def encrypt_id(id_):
    f = Fernet(current_app.config['FERNET_KEY'])
    return f.encrypt(str(id_).encode()).decode()

def decrypt_id(encrypted_id):
    try:
        f = Fernet(current_app.config['FERNET_KEY'])
        return int(f.decrypt(encrypted_id.encode()).decode())
    except:
        return None