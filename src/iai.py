def setup():
    import interpretableai

    interpretableai.install_julia()
    interpretableai.install_system_image()


if __name__ == "__main__":
    setup()
