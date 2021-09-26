import timeit

def foo(text):
    return text


# print(timeit.timeit("foo(a + '\\n')", setup="a = 'hello'", globals=globals()))
# print(timeit.timeit("foo(f'{a}\\n')", setup="a = 'hello'", globals=globals()))
# print(timeit.timeit("foo(a); foo('\\n')", setup="a = 'hello'", globals=globals()))


# print(timeit.timeit('foo("1,2" + ",".join(["a", "b", "c"]))', globals=globals()))
# print(timeit.timeit('foo(",".join(["1", "2"] + ["a", "b", "c"]))', globals=globals()))

class Bar:

    def foo(self):
        print("hello")

    def bar(self):
        print("world")

Bar().foo.__self__.bar()
Bar().foo.__func__(1)