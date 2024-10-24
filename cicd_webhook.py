import requests

def trigger_webhook():
    url = "https://hook.us2.make.com/t7urq2yybrhc57fcd6ccbwzbbxujr3d7"  # 여기에 실제 webhook URL을 입력하세요.
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "diff": "Webhook triggered by GitLab CI/CD pipeline test!!!!",
        "summary": "Webhook triggered by GitLab CI/CD pipeline test!!!!"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print("Webhook successfully triggered!")
        else:
            print(f"Failed to trigger webhook: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    trigger_webhook()