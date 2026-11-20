"""Prepare/deploy one dedicated EC2 benchmark worker and fetch its metrics.

Uses the user's AWS CLI SSO profile locally; no credentials enter the bundle.
All mutated resources are new and named/tagged for the recorded run.
"""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import tarfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT="results/lstm_size_sweep/full/9aab4963bc37dcea/checkpoints/h256_l3_s0.pt"


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()


def aws(spec,*argv):
    command=["aws","--profile",spec["profile"],"--region",spec["region"],"--output","json",
             "--cli-connect-timeout","10","--cli-read-timeout","60",*argv]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-3000:])
    return json.loads(result.stdout) if result.stdout.strip() else {}


def prepare(folder,profile,region):
    folder=folder.resolve()
    if (folder/"manifest.json").exists():raise ValueError("Use a new output folder")
    run="qkp-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]
    spec={"run_id":run,"profile":profile,"region":region,"instance_type":"m7i.2xlarge",
          "volume_gb":400,"max_compute_hours":24,"output":str(folder),"status":"prepared",
          "compute_usd_per_hour":.4032,"resources":{}}
    spec["account"]=aws(spec,"sts","get-caller-identity")["Account"]
    folder.mkdir(parents=True,exist_ok=True)
    files=set()
    for package in ["quantum","simulator","decoders","experiments","training","rl","tests","ops/ec2"]:
        files.update(p for p in (ROOT/package).rglob("*.py") if p.is_file() and not p.is_symlink())
    files.update((ROOT/"docs").glob("*.md"))
    files.update(ROOT/p for p in ["AGENTS.md","RESEARCH_NOTES.md","requirements.txt",CHECKPOINT])
    pins={name:version(name) for name in ["numpy","scipy","qutip","torch","matplotlib","pytest"]}
    spec["pinned_packages"]=pins;spec["python"]="3.13.13"
    archive=folder/"source.tgz"
    with tarfile.open(archive,"w:gz") as tar:
        for path in sorted(files):
            if not path.resolve().is_relative_to(ROOT):raise ValueError("Source outside repository")
            tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    spec["source_sha256"]=digest(archive);spec["source_bytes"]=archive.stat().st_size
    spec["source_manifest"]={str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}
    spec["bucket"]=f'qkp-bench-{spec["account"]}-{run[4:]}'
    save(folder/"manifest.json",spec)
    return spec


def deploy(folder):
    folder=folder.resolve();path=folder/"manifest.json";spec=json.loads(path.read_text())
    if spec["status"]!="prepared":raise ValueError("Manifest is already deployed/partially deployed; inspect it before retry")
    if aws(spec,"sts","get-caller-identity")["Account"]!=spec["account"]:raise ValueError("AWS account changed")
    if digest(folder/"source.tgz")!=spec["source_sha256"]:raise ValueError("Source bundle changed")
    for rel,expected in spec["source_manifest"].items():
        if digest(ROOT/rel)!=expected:raise ValueError("Working source changed after packaging")
    run=spec["run_id"];bucket=spec["bucket"];tags=[{"Key":"Project","Value":"qkp-benchmark"},{"Key":"Owner","Value":"christopher-babayan"},{"Key":"RunId","Value":run}]
    def record(key,value):
        spec["resources"][key]=value;save(path,spec)
    vpcs=aws(spec,"ec2","describe-vpcs","--filters","Name=is-default,Values=true")["Vpcs"]
    if not vpcs:raise ValueError("No default VPC; select an approved subnet explicitly")
    vpc=vpcs[0]["VpcId"]
    subnets=aws(spec,"ec2","describe-subnets","--filters",f"Name=vpc-id,Values={vpc}","Name=default-for-az,Values=true")["Subnets"]
    subnet=sorted([s for s in subnets if s.get("MapPublicIpOnLaunch") and s["AvailableIpAddressCount"]>4],key=lambda s:s["AvailabilityZone"])[0]
    ami=aws(spec,"ssm","get-parameter","--name","/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64")["Parameter"]["Value"]
    image=aws(spec,"ec2","describe-images","--image-ids",ami)["Images"][0]
    spec["status"]="creating_dedicated_resources";save(path,spec)
    args=["s3api","create-bucket","--bucket",bucket]
    if spec["region"]!="us-east-1":args+=["--create-bucket-configuration",json.dumps({"LocationConstraint":spec["region"]})]
    aws(spec,*args);record("bucket",bucket)
    aws(spec,"s3api","put-public-access-block","--bucket",bucket,"--public-access-block-configuration",
        json.dumps({"BlockPublicAcls":True,"IgnorePublicAcls":True,"BlockPublicPolicy":True,"RestrictPublicBuckets":True}))
    aws(spec,"s3api","put-bucket-encryption","--bucket",bucket,"--server-side-encryption-configuration",
        json.dumps({"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}))
    aws(spec,"s3api","put-bucket-tagging","--bucket",bucket,"--tagging",json.dumps({"TagSet":tags}))
    role=run+"-role"
    trust={"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
    role_data=aws(spec,"iam","create-role","--role-name",role,"--assume-role-policy-document",json.dumps(trust),"--tags",json.dumps(tags))["Role"]
    record("role_name",role);record("role_arn",role_data["Arn"])
    aws(spec,"iam","attach-role-policy","--role-name",role,"--policy-arn","arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    policy={"Version":"2012-10-17","Statement":[
        {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":[f"arn:aws:s3:::{bucket}"],"Condition":{"StringLike":{"s3:prefix":[run,run+"/*"]}}},
        {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:AbortMultipartUpload"],"Resource":[f"arn:aws:s3:::{bucket}/{run}/*"]}]}
    aws(spec,"iam","put-role-policy","--role-name",role,"--policy-name","QKPBenchmarkArtifacts","--policy-document",json.dumps(policy))
    aws(spec,"iam","create-instance-profile","--instance-profile-name",role,"--tags",json.dumps(tags))
    record("instance_profile",role)
    aws(spec,"iam","add-role-to-instance-profile","--instance-profile-name",role,"--role-name",role)
    sg=aws(spec,"ec2","create-security-group","--group-name",run,"--description","Isolated QKP benchmark; SSM only, no inbound ports","--vpc-id",vpc)["GroupId"]
    record("security_group",sg)
    aws(spec,"ec2","create-tags","--resources",sg,"--tags",json.dumps(tags))
    aws(spec,"s3","cp",str(folder/"source.tgz"),f"s3://{bucket}/{run}/source/source.tgz","--no-progress","--only-show-errors")
    save(folder/"remote_request.json",{k:spec[k] for k in ["run_id","region","instance_type","max_compute_hours","pinned_packages","python","source_sha256"]})
    aws(spec,"s3","cp",str(folder/"remote_request.json"),f"s3://{bucket}/{run}/source/request.json","--no-progress","--only-show-errors")
    pins=spec["pinned_packages"]
    packages=" ".join(f"{name}=={pins[name]}" for name in ["numpy","scipy","qutip","matplotlib","pytest"])
    bootstrap=f"""#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/qkp-bootstrap.log) 2>&1
export AWS_DEFAULT_REGION={spec['region']}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
qkp_error() {{
  code=$?
  trap - ERR
  set +e
  printf '%s\\n' '{{"status":"bootstrap_failed"}}' > /tmp/qkp-bootstrap-status.json
  aws s3 cp /tmp/qkp-bootstrap-status.json s3://{bucket}/{run}/metrics/status.json --only-show-errors
  aws s3 cp /var/log/qkp-bootstrap.log s3://{bucket}/{run}/metrics/bootstrap.log --only-show-errors
  shutdown -h now
  exit "$code"
}}
trap qkp_error ERR
shutdown -h +1440
systemctl enable --now amazon-ssm-agent
dnf install -y python3-pip
python3 -m pip install uv==0.11.21
mkdir -p /opt/qkp
aws s3 cp s3://{bucket}/{run}/source/source.tgz /opt/qkp/source.tgz --only-show-errors
echo '{spec['source_sha256']}  /opt/qkp/source.tgz' | sha256sum -c -
tar xzf /opt/qkp/source.tgz -C /opt/qkp
cd /opt/qkp
uv python install {spec['python']}
uv venv --python {spec['python']} .venv
uv pip install --python .venv/bin/python {packages}
uv pip install --python .venv/bin/python 'torch=={pins['torch']}' --index-url https://download.pytorch.org/whl/cpu
exec .venv/bin/python -u ops/ec2/worker.py --run-id {run} --bucket {bucket}
"""
    (folder/"bootstrap.sh").write_text(bootstrap)
    subprocess.run(["bash","-n",str(folder/"bootstrap.sh")],check=True)
    launch={"ImageId":ami,"InstanceType":spec["instance_type"],"MinCount":1,"MaxCount":1,"ClientToken":run,
            "IamInstanceProfile":{"Name":role},"InstanceInitiatedShutdownBehavior":"stop",
            "MetadataOptions":{"HttpEndpoint":"enabled","HttpTokens":"required"},
            "NetworkInterfaces":[{"DeviceIndex":0,"SubnetId":subnet["SubnetId"],"Groups":[sg],"AssociatePublicIpAddress":True,"DeleteOnTermination":True}],
            "BlockDeviceMappings":[{"DeviceName":image["RootDeviceName"],"Ebs":{"VolumeSize":spec["volume_gb"],"VolumeType":"gp3","Encrypted":True,"DeleteOnTermination":True}}],
            "TagSpecifications":[{"ResourceType":"instance","Tags":[*tags,{"Key":"Name","Value":run}]},{"ResourceType":"volume","Tags":tags}],
            "UserData":base64.b64encode(bootstrap.encode()).decode()}
    save(folder/"launch.json",launch)
    # IAM instance-profile propagation can lag the successful API response.
    for attempt in range(12):
        try:
            instance=aws(spec,"ec2","run-instances","--cli-input-json","file://"+str(folder/"launch.json"))["Instances"][0]
            break
        except RuntimeError as error:
            if "Invalid IAM Instance Profile" not in str(error) and "InvalidParameterValue" not in str(error):raise
            if attempt==11:raise
            time.sleep(5)
    record("instance_id",instance["InstanceId"])
    spec["status"]="provisioning";spec["launched_utc"]=datetime.now(timezone.utc).isoformat();save(path,spec)
    return spec


def fetch(folder):
    folder=folder.resolve();spec=json.loads((folder/"manifest.json").read_text())
    destination=folder/"metrics";destination.mkdir(exist_ok=True)
    aws(spec,"s3","sync",f's3://{spec["bucket"]}/{spec["run_id"]}/metrics/',str(destination),"--no-progress","--only-show-errors")
    return destination


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest="action",required=True)
    p=sub.add_parser("prepare");p.add_argument("folder",type=Path);p.add_argument("--profile",required=True);p.add_argument("--region",default="us-east-1")
    for name in ["deploy","status","fetch","stop"]:
        p=sub.add_parser(name);p.add_argument("folder",type=Path)
    args=ap.parse_args()
    if args.action=="prepare":
        spec=prepare(args.folder,args.profile,args.region)
        print(json.dumps({k:spec[k] for k in ["run_id","account","source_bytes","instance_type","volume_gb","max_compute_hours"]},indent=2))
    elif args.action=="deploy":
        spec=deploy(args.folder);print(json.dumps({"status":spec["status"],"resources":spec["resources"]},indent=2))
    elif args.action=="fetch":print(fetch(args.folder))
    else:
        spec=json.loads((args.folder/"manifest.json").read_text());instance=spec["resources"]["instance_id"]
        if args.action=="stop":
            print(json.dumps(aws(spec,"ec2","stop-instances","--instance-ids",instance),indent=2))
        else:
            data=aws(spec,"ec2","describe-instances","--instance-ids",instance)["Reservations"][0]["Instances"][0]
            managed=aws(spec,"ssm","describe-instance-information","--filters",f"Key=InstanceIds,Values={instance}")["InstanceInformationList"]
            print(json.dumps({"instance_id":instance,"state":data["State"]["Name"],"ssm_status":managed[0]["PingStatus"] if managed else "not_registered"},indent=2))


if __name__=="__main__":main()
