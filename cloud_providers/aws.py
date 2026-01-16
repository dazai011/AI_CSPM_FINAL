import boto3
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

def create_aws_session(aws_access_key_id, aws_secret_access_key, region_name="us-east-1"):
    """
    Create and validate an AWS session using provided credentials.
    Returns boto3.Session on success, None on failure.
    """
    try:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        # Validate by calling a simple API
        ec2 = session.client("ec2", region_name=region_name)
        ec2.describe_regions()  # if creds/perm wrong this will raise
        return session
    except (ClientError, NoCredentialsError, EndpointConnectionError) as e:
        # ClientError covers invalid credentials or permission errors
        # NoCredentialsError is raised if credentials missing
        # EndpointConnectionError for network/region issues
        # We return None so caller can print a friendly message
        # but include the original message for debugging when needed.
        # Do not log secrets.
        # Print a short message here — caller handles user-facing output.
        # For programmatic use return None.
        # (You may log e somewhere secure if needed.)
        return None
    except Exception:
        return None
