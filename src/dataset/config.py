def adjust_config(options=None):
    if options.dataset.startswith('human'):
        options.nkeypoints = 24
        options.state_dim =  66
        options.fps = 10
        if options.dataset.endswith('partial'):
            options.nepoch = 2000
            options.firstdecay = 1200
            options.seconddecay = 1600
        else:
            options.nepoch = 4000
            options.firstdecay = 3000
            options.seconddecay = 3500
    elif options.dataset.startswith('hand'):
        options.nkeypoints = 21
        options.state_dim = 48
        options.fps = 5
        if options.dataset.endswith('partial'):
            options.nepoch = 2000
            options.firstdecay = 1200
            options.seconddecay = 1600
        else:
            options.nepoch = 2000
            options.firstdecay = 1200
            options.seconddecay = 1600
    else:
        raise ValueError("Wrong Dataset Assignment!")

    return options