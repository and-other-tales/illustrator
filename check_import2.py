try:
    import illustrator
    print('Success - illustrator module found')
except ImportError as e:
    print(f'Error: {e}')
