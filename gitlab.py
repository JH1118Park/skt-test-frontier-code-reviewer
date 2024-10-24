import requests
import subprocess

# GitLab 인스턴스 URL 및 개인 액세스 토큰 설정
GITLAB_URL = "https://gitlab.tde.sktelecom.com"
PRIVATE_TOKEN = ""  # 개인 액세스 토큰으로 교체하세요
PROJECT_PATH = "CRD/sdk/engine/nrtc_media_engine"  # 프로젝트 경로


def get_gitlab_branches(repo_url):
    result = subprocess.run(
        ['git', 'ls-remote', '--heads', repo_url],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        branches = [line.split('/')[-1] for line in lines]
        return branches
    else:
        print("Error:", result.stderr)
        return []

# GitLab 프로젝트 ID 가져오기
response = requests.get(
    f"{GITLAB_URL}/api/v4/projects",
    headers={"PRIVATE-TOKEN": PRIVATE_TOKEN},
    params={"search": PROJECT_PATH.split('/')[-1]}
)

project = next((p for p in response.json() if p['path_with_namespace'] == PROJECT_PATH), None)
if not project:
    raise ValueError("프로젝트를 찾을 수 없습니다.")

project_id = project['id']

# Diff 정보 가져오기
response = requests.get(
    f"{GITLAB_URL}/api/v4/projects/{project_id}/repository/compare",
    headers={"PRIVATE-TOKEN": PRIVATE_TOKEN},
    params={"from": "HEAD~4", "to": "HEAD"}
)

if response.status_code == 200:
    diff_data = response.json().get("diffs", [])
    for diff in diff_data:
        print(f"파일: {diff['new_path']}")
        print(diff['diff'])
else:
    print(f"Diff를 가져오는 데 실패했습니다. 상태 코드: {response.status_code}")

repo_url = f"{GITLAB_URL}/{PROJECT_PATH}.git"
branches = get_gitlab_branches(repo_url)
print(branches)


