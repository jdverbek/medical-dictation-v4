"""
Database abstraction layer for Medical Dictation v4.0
PostgreSQL only - simplified and reliable
"""

import os
from urllib.parse import urlparse

def get_db_connection():
    """Get PostgreSQL database connection"""
    database_url = os.environ.get('DATABASE_URL')
    
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    
    try:
        import psycopg
        from psycopg.rows import dict_row
        
        result = urlparse(database_url)
        
        conn = psycopg.connect(
            dbname=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port,
            row_factory=dict_row
        )
        return conn
        
    except ImportError as e:
        raise ImportError(f"psycopg not available: {e}")
        
    except Exception as e:
        raise ConnectionError(f"PostgreSQL connection failed: {e}")

def is_postgresql():
    """Always returns True - we only use PostgreSQL"""
    return True

def init_db():
    """Initialize PostgreSQL database tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # PostgreSQL schema only
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
                improved_report TEXT,
                differential_diagnosis TEXT,
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
        
        conn.commit()
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """Execute a PostgreSQL query with proper error handling"""
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
    """Get the last inserted ID for PostgreSQL"""
    cursor.execute("SELECT LASTVAL()")
    return cursor.fetchone()[0]

