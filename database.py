# database.py

import os
import sqlite3

# Get the absolute path to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# SQLite database file path
db_file = os.path.join(script_dir, 'restaurant_owners_database.db')


def create_connection():
    """Create a database connection to a SQLite database."""
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(f"SQLite connection error: {e}")
        return None


def get_owner_by_username(username):
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM restaurant_owners WHERE username = ?', (username,))
            owner = cursor.fetchone()
            return owner
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            if conn:
                conn.close()


def create_table():
    """Create the restaurant_owners table."""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS restaurant_owners (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    restaurant_name TEXT NOT NULL
                );
            ''')
            conn.commit()
            cursor.close()
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()


def insert_user(username, password, restaurant_name):
    """Insert a new user into the restaurant_owners table."""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO restaurant_owners (username, password, restaurant_name)
                VALUES (?, ?, ?);
            ''', (username, password, restaurant_name))
            conn.commit()
            cursor.close()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            if conn:
                conn.close()


def add_owner(username, password, restaurant_name):
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO restaurant_owners (username, password, restaurant_name)
                VALUES (?, ?, ?)
            ''', (username, password, restaurant_name))
            conn.commit()
            cursor.close()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            if conn:
                conn.close()


def select_all_users():
    """Retrieve all users from the restaurant_owners table."""
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM restaurant_owners;')
            rows = cursor.fetchall()
            cursor.close()
            return rows
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            if conn:
                conn.close()


# Uncomment the line below if you want to create the table when this script is executed
create_table()
