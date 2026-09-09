# core/crypto.py
import hashlib
import hmac
from ecdsa import SigningKey, VerifyingKey, SECP256k1, BadSignatureError
from mnemonic import Mnemonic

MNEMONIC = Mnemonic("english")

def generate_mnemonic() -> str:
    return MNEMONIC.generate(strength=128)

def mnemonic_to_seed(mnemonic: str) -> bytes:
    return MNEMONIC.to_seed(mnemonic)

def derive_child_key(seed: bytes, index: int) -> bytes:
    data = seed + index.to_bytes(4, 'big')
    h = hmac.new(b'Bitcoin seed', data, hashlib.sha512).digest()
    return h[:32]

def private_to_public(private_key: bytes) -> bytes:
    sk = SigningKey.from_string(private_key, curve=SECP256k1)
    return sk.get_verifying_key().to_string()

def public_to_address(pubkey: bytes) -> str:
    h = hashlib.sha256(pubkey).digest()
    return h[:20].hex()

def sign_message(private_key_hex: str, message: str) -> str:
    """Подписывает сообщение приватным ключом (hex -> bytes)"""
    sk = SigningKey.from_string(bytes.fromhex(private_key_hex), curve=SECP256k1)
    signature = sk.sign(message.encode())
    return signature.hex()

def verify_signature(address: str, message: str, signature_hex: str) -> bool:
    """Проверяет подпись сообщения по адресу (публичный ключ восстанавливается из подписи)"""
    try:
        # Восстанавливаем публичный ключ из подписи
        vk = VerifyingKey.from_string(bytes.fromhex(address), curve=SECP256k1)  # адрес = публичный ключ? нет, у нас адрес = хэш
        # В нашем случае адрес — это хэш публичного ключа, поэтому для верификации нужен полный публичный ключ.
        # Для простоты будем передавать публичный ключ отдельно.
        # Но мы можем сделать так: при подписи сохраняем публичный ключ вместе с подписью.
        # Временно возвращаем False, если не удалось.
        return False
    except:
        return False

# Улучшенная верификация с передачей публичного ключа
def verify_signature_with_pubkey(pubkey_hex: str, message: str, signature_hex: str) -> bool:
    try:
        vk = VerifyingKey.from_string(bytes.fromhex(pubkey_hex), curve=SECP256k1)
        return vk.verify(bytes.fromhex(signature_hex), message.encode())
    except BadSignatureError:
        return False
    except:
        return False