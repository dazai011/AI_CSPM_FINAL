from azure.identity import ClientSecretCredential, AuthenticationError
from azure.mgmt.resource import ResourceManagementClient

def create_azure_client(tenant_id, client_id, client_secret, subscription_id):
    """
    Create Azure ResourceManagement client using provided credentials.
    Validates by listing resource groups.
    Returns ResourceManagementClient on success, None on failure.
    """
    try:
        cred = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)
        client = ResourceManagementClient(credential=cred, subscription_id=subscription_id)
        # Test by listing resource groups (small call)
        _ = list(client.resource_groups.list())
        return client
    except AuthenticationError:
        return None
    except Exception:
        # Covers invalid subscription id, network errors, permission issues, etc.
        return None
