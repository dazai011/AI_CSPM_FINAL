# ai/aws_analyzer.py
from ai.model.rule_engine import analyze_aws_metadata
from ai.model.ai_engine import analyze_with_llm
from ai.model.analyze_utils import redact_metadata, save_report

# Note: do NOT instantiate LocalLLM at import time — that would load large models.


def analyze_aws(session, save: bool = False):
    """
    session: boto3.Session created by your cloud_providers.aws.create_aws_session
    Returns: report dict, or (report, path) if save=True
    """
    ec2 = session.client("ec2")
    s3  = session.client("s3")
    iam = session.client("iam")

    # --- Collect metadata (best-effort, keep read-only) ---
    security_groups = ec2.describe_security_groups()
    instances = ec2.describe_instances()
    buckets = s3.list_buckets()

    # Evaluate bucket ACL / public access (best-effort)
    bucket_public_map = {}
    for b in buckets.get("Buckets", []) if isinstance(buckets, dict) else []:
        name = b.get("Name")
        is_public = False
        pab_cfg = None
        try:
            acl = s3.get_bucket_acl(Bucket=name)
            for g in acl.get("Grants", []):
                uri = g.get("Grantee", {}).get("URI", "")
                if uri and ("AllUsers" in uri or "AuthenticatedUsers" in uri):
                    is_public = True
        except Exception:
            # permission or other error; leave unknown
            pass
        try:
            pab = s3.get_public_access_block(Bucket=name)
            pab_cfg = pab.get("PublicAccessBlockConfiguration")
        except Exception:
            pab_cfg = None
        bucket_public_map[name] = {"is_public": is_public, "public_access_block": pab_cfg}

    # IAM inline policies (best-effort)
    iam_policies = []
    try:
        roles = iam.list_roles().get("Roles", [])[:25]
        for r in roles:
            rn = r.get("RoleName")
            try:
                pnames = iam.list_role_policies(RoleName=rn).get("PolicyNames", [])
                for pn in pnames:
                    doc = iam.get_role_policy(RoleName=rn, PolicyName=pn).get("PolicyDocument")
                    iam_policies.append({"PolicyName": f"{rn}:{pn}", "PolicyDocument": doc})
            except Exception:
                pass
    except Exception:
        pass

    metadata = {
        "security_groups": security_groups,
        "instances": instances,
        "buckets": buckets,
        "bucket_public_map": bucket_public_map,
        "iam_policies": iam_policies
    }

    # --- Rule engine evidence ---
    evidence = analyze_aws_metadata(metadata)

    # Small redacted sample to send to LLM (avoid sending full dumps)
    sample = {"evidence_count": len(evidence), "buckets_sample": list(bucket_public_map.items())[:5]}
    report = analyze_with_llm(evidence, sample)

    # attach audit details
    report.setdefault("raw_evidence", evidence)
    report.setdefault("redacted_sample", redact_metadata(sample))

    if save:
        path = save_report(report)
        return report, path
    return report
