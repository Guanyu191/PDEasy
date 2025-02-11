def elapsed_time(start_time, end_time):
    seconds = end_time - start_time
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    # print(f'used time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s')
    return hours, minutes, seconds