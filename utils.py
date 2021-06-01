from rich.progress import *
import numpy as np

def epsilon_random(epsilon):
    return np.random.random() < epsilon

def pbar(spinner=True, description=True, bar=True, percentage=True, completion=True, time_elapsed = True, time_remaining=True, filesize=False, total_filesize=False, count=False, units='items'):
    progress = []

    if spinner:
        progress.append(SpinnerColumn())

    if description:
        progress.append("[progress.description]{task.description}")

    if bar:
        progress.append(BarColumn())

    if percentage:
        progress.append("[progress.percentage]{task.percentage:>3.0f}%")

    if completion:
        progress.append("{task.completed} of {task.total}")

    if time_elapsed:
        progress.append(TimeElapsedColumn())

    if time_remaining:
        progress.append(TimeRemainingColumn())

    if filesize:
        progress.append(FileSizeColumn())

    if total_filesize:
        progress.extend(['of', TotalFileSizeColumn()])

    if count:
        progress.append("[progress.completed]{task.completed} %s" % units)

    return Progress(*progress)
