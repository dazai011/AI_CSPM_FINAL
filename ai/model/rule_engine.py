# ai/model/rule_engine.py
from typing import List, Dict, Any

def analyze_aws_metadata(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Input metadata keys expected: security_groups, instances, buckets, bucket_public_map, iam_policies
    Returns list of findings: {"issue","resource","severity","detail"}
    """
    findings = []

    # --- Security groups ---
    sgs = metadata.get("security_groups", {}).get("SecurityGroups", []) \
          if isinstance(metadata.get("security_groups"), dict) else (metadata.get("security_groups") or [])
    for sg in sgs:
        sg_id = sg.get("GroupId") or sg.get("GroupName") or "unknown-sg"
        for perm in sg.get("IpPermissions", []):
            from_port = perm.get("FromPort")
            to_port = perm.get("ToPort")
            ip_ranges = perm.get("IpRanges", []) or []
            for ipr in ip_ranges:
                cidr = ipr.get("CidrIp")
                if cidr == "0.0.0.0/0":
                    ports = f"{from_port}-{to_port}" if from_port is not None else "all"
                    severity = "high" if (from_port in (22, 3389) or from_port is None) else "medium"
                    findings.append({
                        "issue": "Security group allows 0.0.0.0/0",
                        "resource": sg_id,
                        "severity": severity,
                        "detail": f"Ports: {ports}"
                    })

    # --- S3 buckets ---
    buckets = metadata.get("buckets", {}).get("Buckets", []) \
              if isinstance(metadata.get("buckets"), dict) else (metadata.get("buckets") or [])
    for b in buckets:
        name = b.get("Name") if isinstance(b, dict) else str(b)
        pubinfo = metadata.get("bucket_public_map", {}).get(name)
        if pubinfo:
            if pubinfo.get("is_public"):
                findings.append({
                    "issue": "S3 bucket is public",
                    "resource": name,
                    "severity": "high",
                    "detail": "Bucket ACL or policy exposes bucket to public"
                })
            else:
                pab = pubinfo.get("public_access_block")
                if pab is None:
                    findings.append({
                        "issue": "S3 public access status unknown",
                        "resource": name,
                        "severity": "low",
                        "detail": "PublicAccessBlock configuration not available"
                    })
                else:
                    # warn if any required flag missing
                    cfg = pab or {}
                    if not all(cfg.get(k, False) for k in ["BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets"]):
                        findings.append({
                            "issue": "S3 public access block not fully enabled",
                            "resource": name,
                            "severity": "medium",
                            "detail": f"PublicAccessBlockConfiguration: {cfg}"
                        })
        else:
            findings.append({
                "issue": "S3 bucket found (ACL/policy not checked)",
                "resource": name,
                "severity": "low",
                "detail": "Bucket listed but ACL/policy not fetched; recommend checking"
            })

    # --- IAM policies ---
    iam_policies = metadata.get("iam_policies", []) or []
    for p in iam_policies:
        name = p.get("PolicyName", "inline-policy")
        doc = p.get("PolicyDocument") or {}
        statements = doc.get("Statement", []) if isinstance(doc, dict) else doc
        if isinstance(statements, dict):
            statements = [statements]
        for st in statements or []:
            actions = st.get("Action")
            resources = st.get("Resource")
            if isinstance(actions, str):
                actions = [actions]
            if isinstance(resources, str):
                resources = [resources]
            if actions and ("*" in actions):
                findings.append({
                    "issue": "IAM policy uses wildcard actions",
                    "resource": name,
                    "severity": "high",
                    "detail": f"Actions: {actions}"
                })
            if resources and ("*" in resources):
                findings.append({
                    "issue": "IAM policy uses wildcard resources",
                    "resource": name,
                    "severity": "high",
                    "detail": f"Resources: {resources}"
                })

    # --- EC2 instances: IMDSv2 enforcement and EBS encryption (best-effort) ---
    instances_res = metadata.get("instances", {}).get("Reservations", []) \
                    if isinstance(metadata.get("instances"), dict) else (metadata.get("instances") or [])
    for res in instances_res:
        instances = res.get("Instances", []) if isinstance(res, dict) else []
        for inst in instances:
            iid = inst.get("InstanceId", "unknown-instance")
            metadata_options = inst.get("MetadataOptions", {}) or {}
            if metadata_options and metadata_options.get("HttpTokens") != "required":
                findings.append({
                    "issue": "IMDSv2 not enforced",
                    "resource": iid,
                    "severity": "medium",
                    "detail": "Instance MetadataOptions HttpTokens != 'required'."
                })
            block_devices = inst.get("BlockDeviceMappings", []) or []
            for bd in block_devices:
                ebs = bd.get("Ebs", {}) or {}
                if "Encrypted" in ebs and ebs.get("Encrypted") is False:
                    findings.append({
                        "issue": "EBS volume unencrypted",
                        "resource": iid,
                        "severity": "medium",
                        "detail": f"Block device {bd.get('DeviceName')} unencrypted"
                    })

    return findings
