def lerp(start, end, size):
    """## Linear Interpolation"""
    gap = (end - start) / (size - 1)

    for i in range(size - 1):
        yield start + i * gap
    yield end


def main():
    print(*lerp(0, 10, 5))
    print(*lerp(0, 10, 4))
    print(*lerp(0, 10, 3))

    print(*lerp(10, -10, 5))
    print(*lerp(10, -10, 4))
    print(*lerp(10, -10, 3))

if __name__ == "__main__":
    main()
