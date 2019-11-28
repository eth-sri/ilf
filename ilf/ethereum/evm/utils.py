try:
    from Crypto.Hash import keccak

    def sha3_256(x): return keccak.new(digest_bits=256, data=x).digest()
except (ImportError, ModuleNotFoundError):
    import sha3 as _sha3

    def sha3_256(x): return _sha3.keccak_256(x).digest()


TT256 = 2 ** 256
TT256M1 = 2 ** 256 - 1
TT255 = 2 ** 255
SECP256K1P = 2**256 - 4294968273


def to_signed(i):
    return i if i < TT255 else i - TT256


def bytes_to_int(value):
    return int.from_bytes(value, byteorder='big')


def encode_int32(v):
    return list(v.to_bytes(32, byteorder='big'))


def int_to_big_endian(value: int):
    return value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")


def big_endian_to_int(value: bytes):
    return int.from_bytes(value, "big")


def sha3(seed):
    return sha3_256(to_string(seed))


def to_string(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return bytes(value, 'utf-8')
    if isinstance(value, int):
        return bytes(str(value), 'utf-8')