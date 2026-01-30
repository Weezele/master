"""Entry point for running as a module: python -m dataset_merger"""

from .application import DatasetMergerApp


def main():
    """Main entry point for the script."""
    app = DatasetMergerApp()
    app.run()


if __name__ == "__main__":
    main()
