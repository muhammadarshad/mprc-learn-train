TAU = 256
ORIGIN = 128

def zadd(a: int, b: int) -> int:
    return (int(a) + int(b)) & 0xFF

def zsub(a: int, b: int) -> int:
    return (int(a) - int(b)) & 0xFF

def cdist(a: int, b: int) -> int:
    da = zsub(a, b)
    db = zsub(b, a)
    return da if da < db else db

def sigma2(x: int, y: int) -> int:
    return zadd(x, y)

def delta2(x: int, y: int) -> int:
    return zsub(x, y)

def half_turn_bit(x: int) -> int:
    return (int(x) >> 7) & 1
