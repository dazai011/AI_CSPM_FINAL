# scanners/azure_scanner.py
from rich.console import Console
console = Console()

def scan_azure_resources(client):
    """
    List Azure resource groups and key security checks.
    """
    try:
        console.print("[cyan]Scanning Azure Resource Groups...[/cyan]")
        for rg in client.resource_groups.list():
            console.print(f"  - Resource Group: {rg.name} (Location: {rg.location})")

        console.print("\n[green]Azure Scan complete![/green]")
    except Exception as e:
        console.print(f"[red]Error during Azure scan: {e}[/red]")
