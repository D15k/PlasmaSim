import cProfile
import pstats
import runpy

def profile_script(file_path: str):
    with cProfile.Profile() as pr:
        runpy.run_path(file_path, run_name='__main__')  # runs it like a script

    stats = pstats.Stats(pr)
    stats.sort_stats('tottime').print_stats(20)


if __name__ == '__main__':
    profile_script(r'Performence_Test\Test-Sim_LD.py')