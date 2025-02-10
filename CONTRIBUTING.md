# Contributing to Granite Snack Cookbook

Thank you for your interest in contributing to the Granite Snack Cookbook! We welcome contributions from the community to help improve and expand this project. Please take a moment to review the guidelines below before you start contributing.

## How to Contribute

### Reporting Issues

If you encounter any issues or have suggestions for improvements, please open an issue on the [GitHub repository](https://github.com/ibm-granite-community/granite-snack-cookbook/issues). Provide as much detail as possible, including steps to reproduce the issue and any relevant screenshots or logs.

### Submitting Pull Requests

1. **Fork the Repository**: Start by forking the repository to your GitHub account.

2. **Clone the Repository**: Clone the forked repository to your local machine.
   ```bash
   git clone https://github.com/your-username/granite-snack-cookbook.git
   cd granite-snack-cookbook
   ```

3. **Create a Branch**: Create a new branch for your feature or bug fix.
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**: Make your changes to the codebase. Ensure that your code follows the project's coding style and best practices.

5. **Commit Changes**: Commit your changes with a descriptive commit message.
   ```bash
   git add .
   git commit -m "Add your commit message here"
   ```

6. **Push Changes**: Push your changes to your forked repository.
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**: Open a pull request on the main repository. Provide a clear and detailed description of your changes.

### Code Style and Formatting

To maintain a consistent code style across the project, please follow these guidelines:

- Use meaningful variable and function names.
- Add comments and docstrings to explain the purpose and functionality of the code.
- Ensure proper indentation, spacing, and line breaks.
- Use tools like `black` or `prettier` to automatically format the code.
- Use linting tools like `flake8` or `pylint` to check for code style issues.

### Setting Up the Development Environment

To set up the development environment, follow these steps:

1. **Python Version**: Ensure you have Python 3.10 or 3.11 installed. You can download it from [python.org](https://www.python.org/downloads/).

2. **Virtual Environment**: It is recommended to create a virtual environment to manage dependencies. You can create a virtual environment using the following command:
   ```bash
   python -m venv granite-env
   ```

3. **Activate Virtual Environment**:
   - On Windows:
     ```bash
     .\granite-env\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source granite-env/bin/activate
     ```

4. **Install Dependencies**: Install the necessary dependencies using the following command:
   ```bash
   pip install -r requirements.txt
   ```

### Running Tests

To ensure the quality of the codebase, please run tests before submitting a pull request. Follow these steps to run the tests:

1. **Install Test Dependencies**: Install the test dependencies using the following command:
   ```bash
   pip install -r requirements-test.txt
   ```

2. **Run Tests**: Run the tests using the following command:
   ```bash
   pytest
   ```

3. **Check Code Coverage**: Check the code coverage using the following command:
   ```bash
   pytest --cov=.
   ```

### Code of Conduct

Please note that this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report any unacceptable behavior to the project maintainers.

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to the Granite Snack Cookbook!
