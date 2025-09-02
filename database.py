"""
Database abstraction layer for Medical Dictation v4.0
Supports both PostgreSQL (cloud) and SQLite (local development)
"""

import os
import sqlite3
from urllib.parse import urlparse

def get_db_connection():
    """Get database connection (PostgreSQL on Render, SQLite locally)"""
    database_url = os.environ.get('DATABASE_URL')
    
    print(f"🔍 DEBUG: DATABASE_URL present: {bool(database_url)}")
    if database_url:
        print(f"🔍 DEBUG: DATABASE_URL starts with: {database_url[:20]}...")
    
    if database_url:
        # PostgreSQL on Render
        try:
            print(f"🔍 DEBUG: Attempting to import psycopg2...")
            import psycopg2
            from psycopg2.extras import RealDictCursor
            print(f"✅ DEBUG: psycopg2 imported successfully")
            
            print(f"🔍 DEBUG: Parsing DATABASE_URL...")
            result = urlparse(database_url)
            print(f"🔍 DEBUG: Database: {result.path[1:]}")
            print(f"🔍 DEBUG: Host: {result.hostname}")
            print(f"🔍 DEBUG: Port: {result.port}")
            print(f"🔍 DEBUG: User: {result.username}")
            
            print(f"🔍 DEBUG: Attempting PostgreSQL connection...")
            conn = psycopg2.connect(
                database=result.path[1:],
                user=result.username,
                password=result.password,
                host=result.hostname,
                port=result.port
            )
            print(f"✅ DEBUG: Connected to PostgreSQL database successfully!")
            return conn
            
        except ImportError as e:
            print(f"❌ DEBUG: psycopg2 import failed: {e}")
            print("🔄 DEBUG: Falling back to SQLite")
            # Fall back to SQLite if psycopg2 not available
            conn = sqlite3.connect('medical_app_v4.db')
            conn.row_factory = sqlite3.Row
            return conn
            
        except Exception as e:
            print(f"❌ DEBUG: PostgreSQL connection failed: {e}")
            print(f"🔍 DEBUG: Error type: {type(e)}")
            import traceback
            print(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
            print("🔄 DEBUG: Falling back to SQLite")
            # Fall back to SQLite if PostgreSQL connection fails
            conn = sqlite3.connect('medical_app_v4.db')
            conn.row_factory = sqlite3.Row
            return conn
    else:
        # SQLite for local development
        print(f"🔍 DEBUG: No DATABASE_URL, using SQLite")
        conn = sqlite3.connect('medical_app_v4.db')
        conn.row_factory = sqlite3.Row
        return conn

def is_postgresql():
    """Check if we're using PostgreSQL"""
    return bool(os.environ.get('DATABASE_URL'))

def init_db():
    """Initialize database tables (works for both PostgreSQL and SQLite)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if is_postgresql():
            print("🔍 DEBUG: Initializing PostgreSQL schema...")
            # PostgreSQL schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT true,
                    consent_given BOOLEAN DEFAULT false,
                    consent_date TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcription_history (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    patient_id TEXT,
                    verslag_type TEXT NOT NULL,
                    original_transcript TEXT,
                    structured_report TEXT,
                    enhanced_transcript TEXT,
                    quality_feedback TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id SERIAL PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    user_id INTEGER,
                    ip_address TEXT,
                    user_agent TEXT,
                    details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            print("🔍 DEBUG: Initializing SQLite schema...")
            # SQLite schema
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
                    updated_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
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
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """Execute a database query with proper error handling"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
        else:
            result = cursor.rowcount
        
        conn.commit()
        return result
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Database query failed: {e}")
        print(f"🔍 Query: {query}")
        print(f"🔍 Params: {params}")
        raise
    finally:
        conn.close()

def get_last_insert_id(conn, cursor):
    """Get the last inserted ID (works for both PostgreSQL and SQLite)"""
    if is_postgresql():
        cursor.execute("SELECT LASTVAL()")
        return cursor.fetchone()[0]
    else:
        return cursor.lastrowid

