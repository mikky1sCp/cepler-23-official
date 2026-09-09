# core/api.py
import requests
from config import API_URL

class NodeAPI:
    @staticmethod
    def get_balance(address: str) -> int:
        try:
            resp = requests.get(f"{API_URL}/balance/{address}", timeout=5)
            if resp.status_code == 200:
                return resp.json().get('available', 0)
            return 0
        except:
            return 0

    @staticmethod
    def send_transaction(from_addr: str, to_addr: str, amount: int, private_key: str) -> dict:
        payload = {
            "from": from_addr,
            "to_pubkey": to_addr,
            "amount": amount,
            "private_key": private_key
        }
        try:
            resp = requests.post(f"{API_URL}/transaction", json=payload, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_history(address: str) -> list:
        try:
            resp = requests.get(f"{API_URL}/history/{address}", timeout=5)
            if resp.status_code == 200:
                return resp.json().get('history', [])
            return []
        except:
            return []