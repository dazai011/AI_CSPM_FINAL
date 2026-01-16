#!/usr/bin/env python3

import cmd2
from rich.console import Console
from rich.panel import Panel
import getpass
import traceback
from typing import List, Dict, Any, Optional

# Optional analyzer helper (your module)
from ai.aws_analyze import analyze_aws

# Import the engine wrapper functions from your ai module
try:
    # keep names consistent with your ai/ai_engine.py
    from ai.ai_engine import get_engine, generate_from_messages, analyze_findings_and_report, AIEngine
except Exception as e:
    get_engine = None
    generate_from_messages = None
    analyze_findings_and_report = None
    AIEngine = None
    LOG_IMPORT_AI_ERR = e

console = Console()


class CSPMShell(cmd2.Cmd):
    # default prompt (no provider connected)
    prompt = "cspm> "
    intro = ""  # banner printed in preloop()

    def __init__(self):
        super().__init__()
        # store clients and active provider
        self.aws_client = None
        self.azure_client = None
        self.gcp_client = None
        self.active_provider = None  # one of: None, "aws", "azure", "gcp"

        # LLM engine (lazy)
        self._ai_engine: Optional[AIEngine] = None

    def preloop(self):
        banner_text = r"""
 @@@@@@@  @@@        @@@@@@   @@@  @@@  @@@@@@@       @@@@@@   @@@@@@@@   @@@@@@@     
@@@@@@@@  @@@       @@@@@@@@  @@@  @@@  @@@@@@@@     @@@@@@@   @@@@@@@@  @@@@@@@@     
!@@       @@!       @@!  @@@  @@!  @@@  @@!  @@@     !@@       @@!       !@@          
!@!       !@!       !@!  !@!  !@!  !@!  !@!  @!@     !@!       !@!       !@!          
!@!       @!!       @!@  !@!  @!@  !@!  @!@  !@!     !!@@!!    @!!!:!    !@!          
!!!       !!!       !@!  !!!  !@!  !!!  !@!  !!!      !!@!!!   !!!!!:    !!!          
:!!       !!:       !!:  !!!  !!:  !!!  !!:  !!!          !:!  !!:       :!!          
:!:        :!:      :!:  !:!  :!:  !:!  :!:  !:!         !:!   :!:       :!:          
 ::: :::   :: ::::  ::::: ::  ::::: ::   :::: ::     :::: ::    :: ::::   ::: :::     
 :: :: :  : :: : :   : :  :    : :  :   :: :  :      :: : :    : :: ::    :: :: :     
  CSPM Security Console (demo)
"""
        console.print(Panel.fit("[bold cyan]Welcome to CSPM Shell[/bold cyan]\nUse 'connect' to authenticate to a cloud provider.", subtitle="CSPM", style="cyan"))
        console.print(banner_text)

    # helper to update prompt based on active provider
    def update_prompt(self):
        if self.active_provider:
            # provider name in red
            self.prompt = f"cspm \033[31m{self.active_provider}\033[0m> "
        else:
            self.prompt = "cspm> "

    # lazy init for AI engine
    def _ensure_ai(self) -> Optional[AIEngine]:
        if get_engine is None:
            return None
        if self._ai_engine is None:
            try:
                # get_engine uses defaults from ai/ai_engine.py (model path / prefs)
                self._ai_engine = get_engine()
            except Exception:
                traceback.print_exc()
                self._ai_engine = None
        return self._ai_engine

    # cleanup on exit - close LLM
    def _close_ai(self):
        try:
            if self._ai_engine:
                self._ai_engine.close()
                self._ai_engine = None
        except Exception:
            traceback.print_exc()

    # -------- Exit/quit commands --------
    def do_exit(self, arg):
        if self.active_provider:
            console.print(f"[red]Exiting {self.active_provider} mode. Returning to normal CSPM shell.[/red]")
            self.active_provider = None
            self.update_prompt()
            return False
        else:
            console.print("[bold red]Exiting CSPM Shell...[/bold red]")
            # close LLM if loaded
            self._close_ai()
            return True

    def do_quit(self, arg):
        return self.do_exit(arg)

    # -------- Show status command --------
    def do_status(self, arg):
        """Show connection status"""
        lines = []
        lines.append(f"AWS connected: {'yes' if self.aws_client else 'no'}")
        lines.append(f"Azure connected: {'yes' if self.azure_client else 'no'}")
        lines.append(f"GCP connected: {'yes' if self.gcp_client else 'no'}")
        lines.append(f"Active provider: {self.active_provider or 'None'}")
        console.print("\n".join(lines))

    # -------- Disconnect command --------
    def do_disconnect(self, arg):
        """Disconnect from the active provider (or use 'disconnect aws' to disconnect specific provider)."""
        target = arg.strip().lower()
        if not target:
            target = self.active_provider

        if target == "aws":
            self.aws_client = None
            if self.active_provider == "aws":
                self.active_provider = None
                self.update_prompt()
            console.print("[green]Disconnected AWS.[/green]")
            return
        if target == "azure":
            self.azure_client = None
            if self.active_provider == "azure":
                self.active_provider = None
                self.update_prompt()
            console.print("[green]Disconnected Azure.[/green]")
            return
        if target == "gcp":
            self.gcp_client = None
            if self.active_provider == "gcp":
                self.active_provider = None
                self.update_prompt()
            console.print("[green]Disconnected GCP.[/green]")
            return

        console.print("[red]No provider specified or unknown provider. Use 'disconnect aws|azure|gcp' or just 'disconnect' to disconnect active provider.[/red]")

    # -------- Connect command --------
    def do_connect(self, arg):
        """Connect to a cloud provider interactively (credentials entered without echo where appropriate).
        Usage: connect
        You will be asked to choose provider and enter credentials.
        """
        choice = console.input("Select cloud provider:\n1. AWS\n2. Azure\n3. GCP\nEnter choice: ").strip()

        # AWS
        if choice == "1" or choice.lower() in ("aws", "a"):
            console.print("[cyan]AWS selected. You will be prompted for credentials (secret inputs will be hidden).[/cyan]")
            aws_key = getpass.getpass("Enter AWS Access Key ID (input hidden): ")
            aws_secret = getpass.getpass("Enter AWS Secret Access Key: ")
            region = console.input("Enter AWS region (default us-east-1): ").strip() or "us-east-1"

            from cloud_providers.aws import create_aws_session
            session = create_aws_session(aws_key, aws_secret, region)
            if session:
                self.aws_client = session
                self.active_provider = "aws"
                self.update_prompt()
                console.print("[green]AWS session created successfully![/green]")
            else:
                console.print("[bold red]Invalid AWS credentials or unable to connect. Please check and try again.[/bold red]")

        # Azure
        elif choice == "2" or choice.lower() in ("azure", "az"):
            console.print("[cyan]Azure selected. Client secret input will be hidden.[/cyan]")
            tenant_id = console.input("Enter Azure Tenant ID: ").strip()
            client_id = console.input("Enter Azure Client ID: ").strip()
            client_secret = getpass.getpass("Enter Azure Client Secret: ")
            subscription_id = console.input("Enter Azure Subscription ID: ").strip()

            from cloud_providers.azure import create_azure_client
            client = create_azure_client(tenant_id, client_id, client_secret, subscription_id)
            if client:
                self.azure_client = client
                self.active_provider = "azure"
                self.update_prompt()
                console.print("[green]Azure client created successfully![/green]")
            else:
                console.print("[bold red]Invalid Azure credentials or unable to connect. Please check and try again.[/bold red]")

        # GCP
        elif choice == "3" or choice.lower() in ("gcp", "g"):
            console.print("[cyan]GCP selected. Enter the path to the service account JSON file. Input is hidden by default.[/cyan]")
            sa_path = getpass.getpass("Enter path to GCP Service Account JSON (input hidden): ").strip()
            if not sa_path:
                sa_path = console.input("Enter path to GCP Service Account JSON (visible): ").strip()

            from cloud_providers.gcp import create_gcp_client
            client = create_gcp_client(sa_path)
            if client:
                self.gcp_client = client
                self.active_provider = "gcp"
                self.update_prompt()
                console.print("[green]GCP client created successfully![/green]")
            else:
                console.print("[bold red]Invalid GCP service account file or unable to connect. Please check and try again.[/bold red]")

        else:
            console.print("[red]Invalid choice. Enter 1 (AWS), 2 (Azure), or 3 (GCP).[/red]")

    # -------- Scan command --------
    def do_scan(self, arg):
        """Run a scan of cloud resources"""
        if not self.active_provider:
            console.print("[red]No active provider. Connect first using 'connect'.[/red]")
            return

        console.print(f"[bold cyan]Starting scan for {self.active_provider.upper()}...[/bold cyan]")

        if self.active_provider == "aws":
            from scanners.aws_scanner import scan_aws_resources
            scan_aws_resources(self.aws_client)

        elif self.active_provider == "azure":
            from scanners.azure_scanner import scan_azure_resources
            scan_azure_resources(self.azure_client)

        elif self.active_provider == "gcp":
            from scanners.gcp_scanner import scan_gcp_resources
            scan_gcp_resources(self.gcp_client)

    # -------- Analyze command (AWS specific) --------
    def do_analyze(self, arg):
        """Analyze AWS security & generate AI report. Usage: analyze --save"""
        if self.active_provider != "aws":
            console.print("[yellow]AI analysis currently supports AWS only. Connect to AWS first.[/yellow]")
            return

        save_flag = "--save" in arg
        console.print("[cyan]Running AWS AI Analyzer...[/cyan]")

        result = analyze_aws(self.aws_client, save=save_flag)

        if isinstance(result, tuple):
            report, path = result
            console.print(report)
            console.print(f"[green]Saved to {path}[/green]")
        else:
            console.print(result)

    # -------- AI free-form prompt command ----------
    def do_ai(self, arg):
        """
        ai <question>
        Ask the local LLM a question. Example:
            ai summarize the last scan findings
        If no argument is supplied, prompts interactively.
        """
        prompt = arg.strip()
        if not prompt:
            try:
                prompt = input("AI prompt: ").strip()
            except Exception:
                print("No prompt provided.")
                return
            if not prompt:
                print("No prompt provided.")
                return

        engine = self._ensure_ai()
        if engine is None:
            print("AI engine not available. Check ai.ai_engine import and model path.")
            if 'LOG_IMPORT_AI_ERR' in globals():
                print("Import error:", LOG_IMPORT_AI_ERR)
            return

        try:
            print("[AI] generating — this may take a while for large local models...")
            resp = engine.generate_from_messages([{"role": "user", "content": prompt}], max_tokens=512, temperature=0.0)
            # Prefer parsed / text output
            out = resp.get("text") or resp.get("raw") or str(resp)
            print("\n=== AI response ===\n")
            print(out)
            print("\n===================\n")
        except Exception as e:
            print("AI generation failed:", e)
            traceback.print_exc()

    # -------- AI analyze findings command ----------
    def do_ai_analyze(self, arg):
        """
        ai_analyze [aws|azure|gcp]
        Collect lightweight findings from the selected provider and ask the LLM to produce a JSON report.
        If no provider is specified, uses the active provider (self.active_provider).
        """
        target = (arg.strip().lower() or getattr(self, "active_provider", "") or "").lower()
        if not target:
            print("Specify provider: ai_analyze aws|azure|gcp or set active_provider first.")
            return

        engine = self._ensure_ai()
        if engine is None:
            print("AI engine not available. Check ai.ai_engine import and model path.")
            return

        findings: List[Any] = []
        # Try to collect some basic findings per provider; errors are non-fatal.
        try:
            if target == "aws":
                aws_session = getattr(self, "aws_client", None)
                if not aws_session:
                    print("No AWS session found (self.aws_client). Connect first.")
                else:
                    try:
                        s3 = aws_session.client("s3")
                        buckets = s3.list_buckets().get("Buckets", [])
                        findings.append({"type": "s3_buckets_count", "count": len(buckets)})
                        for b in buckets[:10]:
                            findings.append({"type": "s3_bucket", "name": b.get("Name")})
                    except Exception as e:
                        findings.append(f"AWS S3 check failed: {e}")

                    try:
                        ec2 = aws_session.client("ec2")
                        sgs = ec2.describe_security_groups().get("SecurityGroups", [])
                        # detect any sg with 0.0.0.0/0 in ingress
                        open_sgs = []
                        for sg in sgs:
                            for perm in sg.get("IpPermissions", []):
                                for r in perm.get("IpRanges", []):
                                    if r.get("CidrIp") == "0.0.0.0/0":
                                        open_sgs.append({"GroupId": sg.get("GroupId"), "GroupName": sg.get("GroupName")})
                                        break
                        findings.append({"type": "open_security_groups", "count": len(open_sgs), "samples": open_sgs[:5]})
                    except Exception as e:
                        findings.append(f"AWS EC2 check failed: {e}")

            elif target == "azure":
                az_rm = getattr(self, "azure_client", None)
                if not az_rm:
                    print("No Azure ResourceManagementClient found (self.azure_client). Connect first.")
                else:
                    try:
                        groups = list(az_rm.resource_groups.list() or [])[:20]
                        findings.append({"type": "azure_resource_group_count", "count": len(groups)})
                        for g in groups[:10]:
                            findings.append({"type": "azure_rg", "name": getattr(g, "name", str(g))})
                    except Exception as e:
                        findings.append(f"Azure RG check failed: {e}")

            elif target == "gcp":
                gcp_client = getattr(self, "gcp_client", None)
                if not gcp_client:
                    print("No GCP client found (self.gcp_client). Connect first.")
                else:
                    try:
                        project = getattr(gcp_client, "project", None) or getattr(gcp_client, "project_id", None)
                        findings.append({"type": "gcp_project", "project": project})
                    except Exception as e:
                        findings.append(f"GCP check failed: {e}")
            else:
                print("Unknown provider. Use aws|azure|gcp.")
        except Exception as e:
            findings.append(f"Collection error: {e}")

        # If we collected nothing, put an explanatory fallback
        if not findings:
            findings.append("No findings collected for provider: " + target)

        # Call your analyzer helper which returns pretty JSON (string) per your ai_engine file
        try:
            print("[AI] sending findings to LLM for structured report...")
            # analyze_findings_and_report returns a string (older compatibility); if None, fallback to engine-based helper
            if analyze_findings_and_report:
                report_text = analyze_findings_and_report(findings)
            else:
                # fallback: call engine directly
                report = engine.generate_from_messages(
                    [{"role": "system", "content": "You are a security auditor assistant. Produce a JSON array named 'report' ... Return ONLY valid JSON."},
                     {"role": "user", "content": "Findings:\n" + "\n".join(str(f) for f in findings)}],
                    max_tokens=512,
                    temperature=0.0,
                )
                report_text = report.get("text", report.get("raw", str(report)))
            print("\n=== AI analysis report ===\n")
            print(report_text)
            print("\n==========================\n")
        except Exception as e:
            print("AI analysis failed:", e)
            traceback.print_exc()


if __name__ == "__main__":
    app = CSPMShell()
    app.cmdloop()
