"""
Authentication System for Medical Dictation v4.0
Secure user management with PostgreSQL/SQLite hybrid support
"""

import os
import hashlib
import secrets
import time
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import session, request, redirect, url_for, flash, jsonify
from database import get_db_connection, is_postgresql

# Configure logging for security audit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_audit.log'),
        logging.StreamHandler()
    ]
)
security_logger = logging.getLogger('security_audit')

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'medical_app_v4.db')

# Rate limiting storage (in production, use Redis)
rate_limit_storage = {}

def init_auth_db():
    """Initialize the authentication database with required tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                consent_given BOOLEAN DEFAULT 0,
                consent_date TIMESTAMP
            )
        ''')
        
        # Create transcription_history table for user's previous outputs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcription_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                patient_id TEXT,
                verslag_type TEXT NOT NULL,
                original_transcript TEXT,
                structured_report TEXT,
                enhanced_transcript TEXT,
                quality_feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create security_events table for audit logging
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                user_id INTEGER,
                ip_address TEXT,
                user_agent TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("🔐 Authentication database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False

def hash_password(password, salt=None):
    """Hash password with salt using SHA-256"""
    if salt is None:
        salt = secrets.token_hex(32)
    
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt

def verify_password(password, stored_hash, salt):
    """Verify password against stored hash"""
    password_hash, _ = hash_password(password, salt)
    return password_hash == stored_hash

def log_security_event(event_type, user_id=None, details=None):
    """Log security events for audit trail"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        user_agent = request.environ.get('HTTP_USER_AGENT', 'unknown')
        
        cursor.execute('''
            INSERT INTO security_events (event_type, user_id, ip_address, user_agent, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (event_type, user_id, ip_address, user_agent, details))
        
        conn.commit()
        conn.close()
        
        # Also log to file
        security_logger.info(f"{event_type} - User: {user_id} - IP: {ip_address} - Details: {details}")
        
    except Exception as e:
        security_logger.error(f"Failed to log security event: {e}")

def rate_limit(max_requests=5, window=300):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get client IP
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
            current_time = datetime.datetime.now()
            
            # Clean old entries
            cutoff_time = current_time - datetime.timedelta(seconds=window)
            if client_ip in rate_limit_storage:
                rate_limit_storage[client_ip] = [
                    timestamp for timestamp in rate_limit_storage[client_ip] 
                    if timestamp > cutoff_time
                ]
            
            # Check rate limit
            if client_ip not in rate_limit_storage:
                rate_limit_storage[client_ip] = []
            
            if len(rate_limit_storage[client_ip]) >= max_requests:
                log_security_event('RATE_LIMIT_EXCEEDED', details=f'IP: {client_ip}')
                flash('Too many attempts. Please try again later.', 'error')
                return redirect(url_for('login'))
            
            # Add current request
            rate_limit_storage[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def create_user(username, email, first_name, last_name, password, consent_given=False):
    """Create a new user account"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username or email already exists
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            conn.close()
            return False, "Username or email already exists"
        
        # Hash password
        password_hash, salt = hash_password(password)
        
        # Insert new user
        cursor.execute('''
            INSERT INTO users (username, email, first_name, last_name, password_hash, salt, consent_given, consent_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, email, first_name, last_name, password_hash, salt, consent_given, 
              datetime.datetime.now() if consent_given else None))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        log_security_event('USER_CREATED', user_id=user_id, details=f'Username: {username}')
        return True, user_id
    except Exception as e:
        return False, str(e)

def authenticate_user(username, password):
    """Authenticate user and return user data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, first_name, last_name, password_hash, salt, is_active
            FROM users WHERE username = ?
        ''', (username,))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "Invalid credentials"
        
        user_id, username, email, first_name, last_name, stored_hash, salt, is_active = user
        
        if not is_active:
            conn.close()
            return False, "Account is disabled"
        
        if verify_password(password, stored_hash, salt):
            # Update last login
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.datetime.now(), user_id))
            conn.commit()
            conn.close()
            
            return True, {
                'id': user_id,
                'username': username,
                'email': email,
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}"
            }
        else:
            conn.close()
            return False, "Invalid credentials"
    except Exception as e:
        return False, str(e)

def login_required(f):
    """Decorator to require authentication for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current user data from session"""
    if 'user_id' in session:
        return {
            'id': session['user_id'],
            'username': session['username'],
            'email': session['email'],
            'first_name': session['first_name'],
            'last_name': session['last_name'],
            'full_name': session['full_name']
        }
    return None

def save_transcription(user_id, verslag_type, original_transcript, structured_report, patient_id=None, enhanced_transcript=None, quality_feedback=None):
    """Save transcription to user's history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transcription_history (user_id, patient_id, verslag_type, original_transcript, structured_report, enhanced_transcript, quality_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, patient_id, verslag_type, original_transcript, structured_report, enhanced_transcript, quality_feedback))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving transcription: {e}")
        return False

def get_user_transcription_history(user_id, limit=50):
    """Get user's transcription history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, patient_id, verslag_type, original_transcript, structured_report, enhanced_transcript, quality_feedback, created_at
            FROM transcription_history 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        history = cursor.fetchall()
        conn.close()
        
        return [{
            'id': row[0],
            'patient_id': row[1],
            'verslag_type': row[2],
            'original_transcript': row[3],
            'structured_report': row[4],
            'enhanced_transcript': row[5],
            'quality_feedback': row[6],
            'created_at': row[7]
        } for row in history]
        
    except Exception as e:
        print(f"Error fetching transcription history: {e}")
        return []

def create_default_admin():
    """Create default admin user if no users exist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM users')
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            # Create default admin user
            success, result = create_user(
                username='admin',
                email='admin@medical-dictation.local',
                first_name='Admin',
                last_name='User',
                password='admin123',  # Change this in production!
                consent_given=True
            )
            
            if success:
                print("🔐 Default admin user created:")
                print("   Username: admin")
                print("   Password: admin123")
                print("   ⚠️  CHANGE PASSWORD IMMEDIATELY IN PRODUCTION!")
                return True
            else:
                print(f"❌ Failed to create default admin: {result}")
                return False
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating default admin: {e}")
        return False

