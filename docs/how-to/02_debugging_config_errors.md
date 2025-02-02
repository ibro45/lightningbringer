# How to Debug Configuration Errors in Lighter

## Introduction to Debugging Configuration Errors

Configuration files are central to Lighter's design, enabling you to define and manage all aspects of your deep learning experiments in a declarative and reproducible manner. However, misconfigurations in your `config.yaml` file are a common source of errors, especially when you are first getting started with Lighter or when you are working with complex configurations.

This how-to guide provides a systematic approach to debugging configuration errors in Lighter, helping you quickly identify and resolve issues in your `config.yaml` files. By following these steps and understanding common error types, you can streamline your debugging process and ensure that your Lighter experiments run smoothly.

## Common Types of Configuration Errors

Configuration errors in Lighter can be broadly categorized into the following types:

1.  **YAML Syntax Errors**: These errors arise from incorrect YAML syntax in your `config.yaml` file, such as:
    *   Incorrect indentation (YAML is indentation-sensitive).
    *   Missing colons, hyphens, or other YAML syntax elements.
    *   Invalid YAML structures (e.g., incorrect nesting of lists or dictionaries).
    *   Unescaped special characters.

    **Example: Common YAML Syntax Errors**

    ```yaml title="config.yaml"
    system: # Missing colon after 'system' - YAML syntax error
      model
        _target_: my_project.models.MyModel # Incorrect indentation
    ```

2.  **Validation Errors**: Lighter uses Cerberus, a powerful validation library, to validate your `config.yaml` file against a predefined schema. Validation errors occur when your configuration does not conform to this schema, such as:
    *   Missing required fields.
    *   Incorrect data types for fields (e.g., string instead of integer).
    *   Invalid values for fields (e.g., value outside allowed range).
    *   Incorrect structure of nested configurations.

    **Example: Common Validation Errors**

    ```yaml title="config.yaml"
    trainer:
      max_epochs: "100" # max_epochs should be an integer, not a string - Validation error

    system:
      optimizer:
        _target_: torch.optim.Adam
        lr: 0.001
      model: # Missing '_target_' for model - Validation error (required field)
        type: my_project.models.MyModel
    ```

3.  **Import Errors**: These errors occur when Lighter fails to dynamically import modules or classes specified in your `config.yaml` using the `_target_` key. Common causes of import errors include:
    *   Incorrect module paths in `_target_` specifications.
    *   Typos in module or class names.
    *   Missing `project` path specification in `config.yaml` when using custom modules.
    *   Issues with your project's Python environment or module dependencies.
    *   Circular dependencies in your custom modules.

    **Example: Common Import Errors**

    ```yaml title="config.yaml"
    project: my_project/

    system:
      model:
        _target_: my_project.models.MyyModel # Typo in class name 'MyyModel' - Import error (class not found)

    trainer:
      callbacks:
        - _target_: lighter.callbacks.MyCallback # Custom callback not in 'lighter.callbacks' - Import error (module not found)
    ```

4.  **Runtime Errors**: Even if your `config.yaml` passes YAML syntax and validation checks, and modules are imported successfully, you may still encounter runtime errors during experiment execution. These errors can arise from:
    *   Incorrect arguments passed to classes or functions in your configuration.
    *   Logical errors in your custom modules' code.
    *   Incompatibility issues between configured components.
    *   Resource limitations (e.g., out-of-memory errors).

    **Example: Common Runtime Errors**

    ```yaml title="config.yaml"
    system:
      model:
        _target_: my_project.models.MyModel
        input_size: "invalid_size" # 'invalid_size' is not a valid integer - Runtime error (invalid argument type)

      optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()" # Incorrect reference - Runtime error (attribute error)
        lr: 0.001
    ```

## Systematic Debugging Steps

To effectively debug configuration errors in Lighter, follow these systematic steps:

1.  **Carefully Read the Error Message**: When Lighter encounters a configuration error, it will typically print an informative error message to the console. **Pay close attention to the entire error message**, as it often contains valuable clues about the type and location of the error.

    *   **Identify the Error Type**: The error message usually indicates the type of error (YAML syntax, validation, import, runtime).
    *   **Locate the Error Source**: The error message may point to the specific line number or section in your `config.yaml` file where the error occurred. It may also indicate the module or class that caused the error.
    *   **Understand the Root Cause**: Try to understand the underlying reason for the error based on the error message. Is it a syntax issue, a validation failure, an import problem, or a runtime issue?

2.  **Validate Your `config.yaml` Syntax**: If you suspect a YAML syntax error, carefully review your `config.yaml` file for common syntax mistakes:

    *   **Indentation**: Ensure consistent and correct indentation (typically 2 spaces) throughout the file.
    *   **Colons and Hyphens**: Check for missing colons after keys in dictionaries and hyphens before items in lists.
    *   **Quotes**: Verify that strings are properly quoted if needed (e.g., for values containing special characters or spaces).
    *   **YAML Linters**: Consider using online YAML linters or VS Code extensions that can automatically detect YAML syntax errors.

3.  **Validate Against the Lighter Configuration Schema**: If the error message indicates a validation error, or if you suspect a validation issue, you need to check your `config.yaml` against the Lighter configuration schema.

    *   **Refer to the `SCHEMA` in `engine/config.py`**: The Lighter configuration schema is defined in the `SCHEMA` variable within the `lighter/engine/config.py` file. **Examine this schema definition** to understand the expected structure, required fields, data types, and allowed values for each configuration section.
    *   **Compare Your `config.yaml` to the `SCHEMA`**: Carefully compare your `config.yaml` file to the `SCHEMA` definition, section by section, field by field. Look for discrepancies in structure, missing required fields, incorrect data types, or invalid values.
    *   **Pay Attention to Nested Configurations**: Validation errors often occur in nested configuration sections. Make sure you understand the expected structure of nested dictionaries and lists as defined in the `SCHEMA`.

4.  **Troubleshoot Import Errors**: If you encounter import errors, systematically troubleshoot the module import process:

    *   **Verify Module Paths**: Double-check the module paths specified in your `_target_` keys in `config.yaml`. Ensure that the paths are correct, relative to your project root directory (if using custom modules) or to the Lighter framework itself (for built-in modules).
    *   **Check for Typos**: Carefully check for typos in module names, class names, and file paths in your `_target_` specifications. Typos are a common source of import errors.
    *   **Ensure `project` Path is Correct**: If you are using custom modules, verify that you have correctly specified the `project` path in your `config.yaml` and that it points to the root directory of your project.
    *   **Verify Module Files Exist**: Make sure that the Python module files you are trying to import actually exist at the specified paths.
    *   **Check Python Environment**: Ensure that your Python environment is correctly set up and that all required dependencies for your custom modules are installed.
    *   **Inspect Error Stack Traces**: If the import error message includes a stack trace, examine the stack trace for more detailed information about the import failure.

5.  **Debug Runtime Errors**: Runtime errors can be more challenging to debug, as they occur during experiment execution rather than during configuration loading or validation.

    *   **Examine Error Stack Traces**: When a runtime error occurs, Lighter will typically print a stack trace. **Carefully examine the stack trace** to pinpoint the exact line of code where the error originated.
    *   **Check Argument Compatibility**: Runtime errors in Lighter configurations often arise from incorrect arguments passed to classes or functions. Verify that the arguments you are providing in your `config.yaml` are compatible with the expected arguments of the target classes or functions. Check argument names, data types, and value ranges.
    *   **Inspect Custom Module Code**: If the runtime error occurs within your custom modules, carefully review your custom module code for logical errors, incorrect variable usage, or other programming mistakes.
    *   **Use a Python Debugger**: For complex runtime errors, consider using a Python debugger (e.g., `pdb`, `ipdb`, or a debugger integrated into your IDE) to step through your code, inspect variables, and understand the program's execution flow. You can set breakpoints in your custom modules or even in Lighter's internal code to debug runtime issues.
    *   **Simplify Your Configuration**: To isolate runtime errors, try simplifying your `config.yaml` file by removing or commenting out sections of the configuration. Gradually add sections back in until the error reappears, helping you narrow down the source of the problem.

6.  **Seek Help and Community Support**: If you are still unable to resolve a configuration error after following these steps, don't hesitate to seek help from the Lighter community:

    *   **Consult the Lighter Documentation**: Review the Lighter documentation, especially the sections related to configuration, for detailed explanations and examples.
    *   **Check the Lighter GitHub Repository**: Explore the Lighter GitHub repository for frequently asked questions, issue discussions, or similar error reports.
    *   **Reach Out to the Lighter Community**: If you are part of a Lighter user community or forum, post your error message and configuration details to seek assistance from other users or Lighter developers.

## Example Scenarios and Solutions

To further illustrate the debugging process, let's consider a few example scenarios of common configuration errors and their solutions:

**Scenario 1: Validation Error - Missing Required Field**

**Error Message**:

```
Cerberus validation error in section 'system.model':
Missing required field: '_target_'
```

**Cause**: The `model` section in the `config.yaml` is missing the required `_target_` field, which specifies the model class to be used.

**Solution**: Add the `_target_` field to the `model` section, specifying the desired model class:

```yaml title="config.yaml"
system:
  model:
    _target_: my_project.models.MyModel # Add '_target_' field
    # ... model arguments ...
```

**Scenario 2: Import Error - Module Not Found**

**Error Message**:

```
ImportError: ModuleNotFoundError: No module named 'my_project.mmodels'
```

**Cause**: There is a typo in the module path `my_project.mmodels` in the `_target_` specification. The correct module path is likely `my_project.models`.

**Solution**: Correct the typo in the module path in the `_target_` specification:

```yaml title="config.yaml"
system:
  model:
    _target_: my_project.models.MyModel # Correct module path to 'my_project.models'
    # ... model arguments ...
```

**Scenario 3: Runtime Error - Invalid Argument Type**

**Error Message**:

```
TypeError: __init__() got an unexpected keyword argument 'input_size'
```

**Cause**: The `MyModel` class constructor (`__init__`) in `my_project.models.MyModel` does not accept an argument named `input_size`, but the `config.yaml` is providing this argument.

**Solution**:

1.  **Inspect the `MyModel` Class**: Examine the `my_project/models/my_model.py` file to check the constructor (`__init__`) definition of the `MyModel` class and identify the correct argument names it accepts.
2.  **Correct Argument Names in `config.yaml`**: Update the `config.yaml` file to use the correct argument names as defined in the `MyModel` class constructor.

    For example, if the `MyModel` class constructor expects `input_dim` instead of `input_size`, update the `config.yaml` as follows:

    ```yaml title="config.yaml"
    system:
      model:
        _target_: my_project.models.MyModel
        input_dim: 784 # Correct argument name to 'input_dim'
        # ... other arguments ...
    ```

## Recap: Mastering Configuration Debugging

Debugging configuration errors is an essential skill for working effectively with Lighter. By understanding common error types, following a systematic debugging approach, and utilizing the tools and techniques outlined in this guide, you can efficiently identify and resolve configuration issues in your `config.yaml` files. This will enable you to streamline your Lighter experiment workflows and focus on the core aspects of your deep learning research and development.

Next, explore the [How-To guide on Creating a Custom Metric](03_creating_a_custom_metric.md) to learn how to extend Lighter with your own evaluation metrics, or return to the [How-To guides section](../how-to/) for more practical problem-solving guides. You can also go back to the [Explanation section](../explanation/) for more conceptual documentation or the [Tutorials section](../tutorials/) for end-to-end examples.
