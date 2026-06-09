"""
Easy way to quickly run the parser with an example
"""

from parse import Parser

def main():
    print("Testing Results Parser")

    parser = Parser('output')

    output_path = parser.parse(['../evaluation-commands/instruction/results'])

    print("Parsing Complete")
    print(f"Output file: {output_path}")

if __name__ == "__main__":
    main()
