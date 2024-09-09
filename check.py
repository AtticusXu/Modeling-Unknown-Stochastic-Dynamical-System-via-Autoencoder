import sys
import subprocess


def check_environment():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    try:
        import tensorflow as tf

        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("TensorFlow is not installed or not found in the current environment.")

    print("\nInstalled packages:")
    subprocess.run([sys.executable, "-m", "pip", "list"])


if __name__ == "__main__":
    check_environment()
