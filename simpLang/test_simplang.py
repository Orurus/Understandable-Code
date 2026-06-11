"""
Test script for SimpLang transpiler.
Runs the transpiler on test code and verifies output.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from transpiler import transpile


def test(description, simp_code, expected_python):
    """Test that transpile produces expected Python."""
    result = transpile(simp_code)
    if result.strip() == expected_python.strip():
        print(f"  PASS: {description}")
        return True
    else:
        print(f"  FAIL: {description}")
        print(f"    Expected:\n{expected_python}")
        print(f"    Got:\n{result}")
        return False


def main():
    print("=" * 50)
    print("  SimpLang Transpiler Tests")
    print("=" * 50)
    print()

    tests_passed = 0
    tests_run = 0

    # 1. Hello World
    tests_run += 1
    if test("say() -> print()",
            'say("Hello")',
            'print("Hello")'):
        tests_passed += 1

    # 2. Variable assignment (passthrough)
    tests_run += 1
    if test("Variable assignment",
            "x = 5",
            "x = 5"):
        tests_passed += 1

    # 3. ask() -> input()
    tests_run += 1
    if test("ask() -> input()",
            'name = ask("Name: ")',
            'name = input("Name: ")'):
        tests_passed += 1

    # 4. if statement
    tests_run += 1
    if test("if statement",
            "if x > 5\n    say('big')\nend",
            "if x > 5:\n    print('big')\npass"):
        tests_passed += 1

    # 5. if-else
    tests_run += 1
    if test("if-else",
            "if x > 5\n    say('big')\nelse\n    say('small')\nend",
            "if x > 5:\n    print('big')\nelse:\n    print('small')\npass"):
        tests_passed += 1

    # 6. loop while
    tests_run += 1
    if test("loop while",
            "loop while x < 10\n    x = x + 1\nend",
            "while x < 10:\n    x = x + 1\npass"):
        tests_passed += 1

    # 7. loop range (to)
    tests_run += 1
    if test("loop range (to)",
            "loop i = 1 to 5\n    say(i)\nend",
            "for i in range(1, (5) + 1):\n    print(i)\npass"):
        tests_passed += 1

    # 8. loop in
    tests_run += 1
    if test("loop in",
            "loop item in list\n    say(item)\nend",
            "for item in list:\n    print(item)\npass"):
        tests_passed += 1

    # 9. fun -> def
    tests_run += 1
    if test("fun -> def",
            "fun greet(name)\n    say('Hi ' + name)\nend",
            "def greet(name):\n    print('Hi ' + name)\npass"):
        tests_passed += 1

    # 10. ret -> return
    tests_run += 1
    if test("ret -> return",
            "ret x + 1",
            "return x + 1"):
        tests_passed += 1

    # 11. Booleans
    tests_run += 1
    if test("yeah/nah/nope",
            "a = yeah\nb = nah\nc = nope",
            "a = True\nb = False\nc = None"):
        tests_passed += 1

    # 12. Comments
    tests_run += 1
    if test("// comments",
            "// this is a comment\nx = 5 // inline",
            "x = 5"):
        tests_passed += 1

    # 13. Type keywords not mangled in convert_to_*
    tests_run += 1
    if test("convert_to_* not mangled",
            'convert_to_txt(5)',
            'convert_to_txt(5)'):
        tests_passed += 1

    # 14. import
    tests_run += 1
    if test("import",
            "import stdlib",
            "import stdlib"):
        tests_passed += 1

    # Print results
    print()
    print("=" * 50)
    print(f"  Results: {tests_passed}/{tests_run} tests passed")
    if tests_passed == tests_run:
        print("  All tests passed!")
    else:
        print(f"  {tests_run - tests_passed} tests failed")
    print("=" * 50)

    return tests_passed == tests_run


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)