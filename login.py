# login.py

from database import get_owner_by_username


def login(username, password):
    """
    Authenticate a user based on their username and password.

    Parameters:
    - username (str): The username of the user.
    - password (str): The password of the user.

    Returns:
    - bool: True if authentication is successful, False otherwise.
    """
    try:
        owner = get_owner_by_username(username)
        if owner and owner[2] == password:  # owner[2] is the hashed password in real-world scenarios
            return True
    except Exception as e:
        # Add appropriate logging or exception handling here
        print(f"Error during login: {e}")

    return False
