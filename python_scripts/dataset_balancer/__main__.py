"""Package entry point for running as module."""

from .application import DatasetBalancerApp


def main():
    """Main entry point."""
    app = DatasetBalancerApp()
    app.run()


if __name__ == "__main__":
    main()
