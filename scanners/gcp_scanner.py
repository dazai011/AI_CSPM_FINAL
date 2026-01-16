# scanners/gcp_scanner.py
from google.cloud import storage, compute_v1
from rich.console import Console
console = Console()

def scan_gcp_resources(client):
    """
    Perform a basic GCP security scan.
    """
    try:
        console.print("[cyan]Scanning GCP Storage Buckets...[/cyan]")
        storage_client = storage.Client(credentials=client._credentials)
        for bucket in storage_client.list_buckets():
            console.print(f"  - Bucket: {bucket.name}")

        console.print("\n[cyan]Scanning GCP Compute Instances...[/cyan]")
        compute_client = compute_v1.InstancesClient(credentials=client._credentials)
        for zone in compute_client.aggregated_list(project=client.project_id):
            instances = zone[1].instances or []
            for instance in instances:
                console.print(f"  - Instance: {instance.name}")

        console.print("\n[green]GCP Scan complete![/green]")
    except Exception as e:
        console.print(f"[red]Error during GCP scan: {e}[/red]")
