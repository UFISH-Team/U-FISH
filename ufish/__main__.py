def main():
    import fire
    from .cli import UFishCLI
    fire.Fire(UFishCLI)


if __name__ == '__main__':
    main()
