from os import path, sys, getcwd

TEST_MODE = True


def change_execution_directory(relative_path_directory_to_execute_from: str):
    ''' Receives a string of a relative path to a directory to execute this
    code from. It's useful for testing libraries in other folders
    :args: relative_path_directory_to_execute_from -> str
    '''
    PACKAGE_PARENT = relative_path_directory_to_execute_from
    SCRIPT_DIR = path.dirname(
        path.realpath(
            path.join(
                getcwd(),
                path.expanduser(__file__)
                        )))
    sys.path.append(path.normpath(path.join(SCRIPT_DIR, PACKAGE_PARENT)))


if TEST_MODE:
    change_execution_directory("../")
