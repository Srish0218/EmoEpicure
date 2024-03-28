import mysql.connector
import os

# Get the absolute path to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# MySQL connection details
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Srish.0211',
    'database': 'epicureglow',
    'raise_on_warnings': True
}

def create_connection():
    """Create a database connection to a MySQL database."""
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as e:
        print(f"MySQL connection error: {e}")
        return None

def get_owner_by_username(username):
    conn = create_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM restaurant_owners WHERE username = %s', (username,))
            owner = cursor.fetchone()
            return owner
        except mysql.connector.Error as e:
            print(f"MySQL error: {e}")
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
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) NOT NULL UNIQUE,
                    password VARCHAR(255) NOT NULL,
                    restaurant_name VARCHAR(255) NOT NULL
                );
            ''')
            conn.commit()
            cursor.close()
        except mysql.connector.Error as e:
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
                VALUES (%s, %s, %s);
            ''', (username, password, restaurant_name))
            conn.commit()
            cursor.close()
        except mysql.connector.Error as e:
            print(f"MySQL error: {e}")
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
                VALUES (%s, %s, %s)
            ''', (username, password, restaurant_name))
            conn.commit()
            cursor.close()
        except mysql.connector.Error as e:
            print(f"MySQL error: {e}")
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
        except mysql.connector.Error as e:
            print(f"MySQL error: {e}")
        finally:
            if conn:
                conn.close()

# Uncomment the line below if you want to create the table when this script is executed
create_table()

if __name__ == "__main__":
    # If this script is run, create the table (initialize the database)
    create_table()
    print("MySQL Database started...")
