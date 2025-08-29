# Contributing to Fake Review Detection System

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## 🤝 How to Contribute

### 1. Fork the Repository
- Click the "Fork" button at the top right of the repository page
- Clone your fork to your local machine

### 2. Set Up Development Environment
```bash
git clone https://github.com/yourusername/fake-review-detection.git
cd fake-review-detection
conda create -n review_classifier python=3.12
conda activate review_classifier
pip install -r requirements.txt
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Write clean, documented code
- Follow Python PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed

### 5. Test Your Changes
```bash
# Run existing tests
python -m pytest tests/

# Test your specific changes
python enhanced_bart_finetune.py  # For training changes
python evaluate_fine_tuned_bart.py  # For evaluation changes
```

### 6. Commit Your Changes
```bash
git add .
git commit -m "Add: Clear description of your changes"
```

### 7. Submit a Pull Request
- Push your branch to your fork
- Create a pull request with a clear title and description
- Reference any related issues

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate

### Documentation
- Update README.md if you add new features
- Add comments for complex logic
- Update docstrings for modified functions

### Testing
- Write tests for new functionality
- Ensure existing tests still pass
- Test with different data sizes and edge cases

## 🐛 Reporting Bugs

When reporting bugs, please include:
- Python version and operating system
- Complete error message and stack trace
- Steps to reproduce the issue
- Expected vs actual behavior

## 💡 Suggesting Enhancements

We welcome suggestions for improvements! Please include:
- Clear description of the enhancement
- Use case and benefits
- Proposed implementation approach (if you have one)

## 🔧 Areas Where We Need Help

- **Model Optimization**: Improving training efficiency and accuracy
- **Data Processing**: Better data preprocessing and augmentation
- **Evaluation Metrics**: Additional evaluation methods
- **Documentation**: Examples, tutorials, and guides
- **Testing**: Unit tests and integration tests
- **Performance**: Speed and memory optimizations

## 📄 Code of Conduct

Please be respectful and constructive in all interactions. We're all here to learn and improve the project together.

## 🙋‍♀️ Questions?

If you have questions about contributing, feel free to:
- Open an issue with the "question" label
- Start a discussion in the Discussions tab
- Contact the maintainers directly

Thank you for contributing! 🚀
