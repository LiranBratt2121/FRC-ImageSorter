import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource

from typing import List

def authenticate_api(api_lib: str, version: str, scopes: List[str], secret_file_path: str, port=4040, host='localhost') -> Resource:
    """
    Authenticates with the Google Photos API and returns the service object, and saves the auth token as a file.
    
    Args:
        api_lib (str): Name of the API library to authenticate.
        version (str): Version of the API.
        scopes (List[str]): List of scopes required for authentication.
        secret_file_path (str): Path to the client secret file.
        port (int): Port to run the local server for authentication. Default is 4040.
        host (str): Hostname for the local server. Default is 'localhost'.
        
    Returns:
        Resource: Service object for the authenticated API.
    """
    # Create flow object.
    flow = InstalledAppFlow.from_client_secrets_file(secret_file_path, scopes)

    # Run the authentication flow.
    credentials = flow.run_local_server(port=port, host=host, redirect_uri_trailing_slash=False)
    
    # Writes token file.
    with open(os.path.join(os.getcwd(), 'google_photos_api', 'config', 'token.json'), 'w') as token:
        token.write(credentials.to_json())

    # Build the service.
    service = build(api_lib, version, credentials=credentials, static_discovery=False)
    
    return service
