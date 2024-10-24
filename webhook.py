import requests

# 파일에서 데이터를 읽기
file_path = '/Users/1111792/Documents/log_mo.txt'  # 출력 파일 경로
with open(file_path, 'r') as file:
    output_data = file.read()

# Make Webhook URL
webhook_url = "https://hook.us2.make.com/t7urq2yybrhc57fcd6ccbwzbbxujr3d7"

# 전송할 데이터 (JSON 형태로 설정)
payload = {
    "data": output_data
}

# 요청 헤더 설정
headers = {
    "Content-Type": "application/json"
}

# POST 요청 전송
response = requests.post(webhook_url, json=payload, headers=headers)

# 응답 확인
if response.status_code == 200:
    print("데이터가 성공적으로 전송되었습니다.")
else:
    print(f"오류 발생: {response.status_code}")
    print(response.text)
