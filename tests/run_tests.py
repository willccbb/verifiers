"""
Test runner script for verifiers package.
"""
import sys
import pytest
from pathlib import Path


def run_tests(test_pattern=None, verbose=False, coverage=False):
    """
    Run the test suite with optional parameters.
    
    Args:
        test_pattern: Specific test pattern to run (e.g., "test_parsers.py::TestParser")
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    args = []
    
    # Set test directory
    test_dir = Path(__file__).parent
    args.append(str(test_dir))
    
    # Add specific test pattern if provided
    if test_pattern:
        args[-1] = str(test_dir / test_pattern)
    
    # Configure verbosity
    if verbose:
        args.append("-v")
    else:
        args.append("-q")
    
    # Configure coverage
    if coverage:
        args.extend([
            "--cov=verifiers",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Add other useful flags
    args.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Strict marker checking
        "-W", "ignore::DeprecationWarning",  # Ignore deprecation warnings
    ])
    
    print(f"Running tests with args: {' '.join(args)}")
    return pytest.main(args)


def run_parser_tests():
    """Run only parser tests."""
    return run_tests("test_parsers.py", verbose=True)


def run_rubric_tests():
    """Run only rubric tests.""" 
    return run_tests("test_rubrics.py", verbose=True)


def run_environment_tests():
    """Run only environment tests."""
    return run_tests("test_environments.py", verbose=True)


def run_all_tests():
    """Run all tests with coverage."""
    return run_tests(verbose=True, coverage=True)


def run_quick_tests():
    """Run all tests quickly without coverage."""
    return run_tests(verbose=False, coverage=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "parsers":
            exit_code = run_parser_tests()
        elif command == "rubrics":
            exit_code = run_rubric_tests()
        elif command == "environments":
            exit_code = run_environment_tests()
        elif command == "all":
            exit_code = run_all_tests()
        elif command == "quick":
            exit_code = run_quick_tests()
        else:
            # Treat as specific test pattern
            exit_code = run_tests(command, verbose=True)
    else:
        # Default: run all tests
        exit_code = run_all_tests()
    
    sys.exit(exit_code)