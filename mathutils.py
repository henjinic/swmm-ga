def lerp(start, end, size):
    """## Linear Interpolation"""
    gap = (end - start) / (size - 1)

    for i in range(size - 1):
        yield start + i * gap
    yield end


def main():
    for i in lerp(0, 10, 5):
        print(i)

if __name__ == "__main__":
    main()
