from google.cloud import resourcemanager_v3
from google.api_core.exceptions import GoogleAPICallError, DefaultCredentialsError
import os

def create_gcp_client(service_account_json_path):
    """
    Create a GCP ProjectsClient from a service account JSON file.
    Validate by listing projects.
    Returns ProjectsClient on success, None on failure.
    """
    try:
        if not os.path.isfile(service_account_json_path):
            return None

        client = resourcemanager_v3.ProjectsClient.from_service_account_file(service_account_json_path)
        # Validate by attempting to list projects (returns generator)
        # We won't iterate all — just attempt to fetch the first page via the iterator call.
        projects = list(client.list_projects(page_size=1))
        return client
    except (DefaultCredentialsError, GoogleAPICallError, FileNotFoundError):
        return None
    except Exception:
        return None
