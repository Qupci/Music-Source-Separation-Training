def encode_bitstring(bitstring: str) -> int:
    """
    Encode a string of '0' and '1' (e.g. '10100') into a bitmask integer.
    Leftmost char = highest bit, rightmost char = lowest bit.
    """
    return int(bitstring, 2)


def decode_bitmask(value: int, length: int) -> str:
    """
    Decode a bitmask integer back into a bitstring of given length.
    Leftmost char = highest bit, rightmost char = lowest bit.
    """
    return format(value, f'0{length}b')


# Example usage:
bitstring = "0111"
encoded = encode_bitstring(bitstring)
decoded = decode_bitmask(encoded, len(bitstring))

value = "46"
bitstring_len = 8
decoded = decode_bitmask(int(value), bitstring_len)
encoded = encode_bitstring(decoded)

print("Bitstring:", bitstring)
print("Encoded :", encoded)
print("Decoded :", decoded)
