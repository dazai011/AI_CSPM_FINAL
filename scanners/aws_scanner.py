# scanners/aws_scanner.py
import boto3
from rich.console import Console
console = Console()

def scan_aws_resources(session):
    """
    Perform a basic AWS security posture scan.
    Enumerates S3, EC2, IAM and checks for misconfigurations.
    """
    try:
        ec2 = session.client("ec2")
        s3 = session.client("s3")
        iam = session.client("iam")

        console.print("[cyan]Scanning EC2 instances...[/cyan]")
        instances = ec2.describe_instances()
        for reservation in instances["Reservations"]:
            for instance in reservation["Instances"]:
                console.print(f"  - EC2 Instance ID: {instance['InstanceId']} ({instance['State']['Name']})")

        console.print("\n[cyan]Scanning S3 buckets...[/cyan]")
        buckets = s3.list_buckets()
        for bucket in buckets["Buckets"]:
            name = bucket["Name"]
            try:
                acl = s3.get_bucket_acl(Bucket=name)
                console.print(f"  - {name}: Public Access = {'Yes' if any(g['Grantee'].get('URI') for g in acl['Grants']) else 'No'}")
            except Exception:
                console.print(f"  - {name}: Unable to retrieve ACL")

        console.print("\n[cyan]Scanning IAM users...[/cyan]")
        users = iam.list_users()
        for user in users["Users"]:
            console.print(f"  - IAM User: {user['UserName']}")

        console.print("\n[green]AWS Scan complete![/green]")

    except Exception as e:
        console.print(f"[red]Error during AWS scan: {e}[/red]")
