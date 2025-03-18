import fire

from typesafe_llm.parser.parser_ts import parse_ts_program


def main(
    input_file: str,
):
    with open(input_file, "r") as f:
        file_content = f.read()
    states = parse_ts_program(file_content, print_failure_point=True)
    print("-----------------------------")
    if states:
        print("Parsed successfully")
    else:
        print("Parsing failed at given point")


if __name__ == "__main__":
    fire.Fire(main)
