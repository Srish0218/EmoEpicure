# signup.py

from database import add_owner


def signup(username, password, restaurant_name):
    """
    Create a new user account and store it in the database.

    Parameters:
    - username (str): The username for the new user.
    - password (str): The password for the new user. (In a real-world scenario, this should be hashed.)
    - restaurant_name (str): The name of the restaurant associated with the new user.
    """
    try:
        add_owner(username, password, restaurant_name)  # In a real-world scenario, password should be hashed
    except Exception as e:
        # Add appropriate logging or exception handling here
        print(f"Error during signup: {e}")
