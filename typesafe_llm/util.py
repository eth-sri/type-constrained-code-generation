def pflush(string: str):
    print(string, flush=True, end="")


def delete(string: str, _pflush=pflush):
    _pflush(len(string) * "\b" + len(string) * " " + len(string) * "\b")
